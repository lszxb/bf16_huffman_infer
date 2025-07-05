#include <torch/all.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define OP_PER_LANE 1

namespace bf16_huffman_infer {

static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}


__global__ void gemv_bf16_kernel(
    const nv_bfloat162* A, const nv_bfloat162* X, nv_bfloat16* Y,
    int M, int N
) {
    N /= 2;

    int thread_id = ((blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) * 2;

    int warp_group_size = warpSize * 2;

    int warp_group_id = thread_id / warp_group_size;
    int lane_id = thread_id % warp_group_size;

    int stride = N;
    
    if (warp_group_id * OP_PER_LANE > M) {
        return; // no work to do
    }

    const int boundry_offset = max((warp_group_id + 1) * OP_PER_LANE - M, 0);

    const std::array<nv_bfloat162, 2> *pa = (const std::array<nv_bfloat162, 2> *)&A[(warp_group_id * OP_PER_LANE - boundry_offset) * stride + lane_id];
    const std::array<nv_bfloat162, 2> *px = (const std::array<nv_bfloat162, 2> *)&X[lane_id];

    std::array<nv_bfloat162, 2> x;
    std::array<nv_bfloat162, 2> a[OP_PER_LANE];
    float y[OP_PER_LANE];

    #pragma unroll
    for (int i = 0; i < OP_PER_LANE; i++) {
        y[i] = 0.0f;
    }

    __syncwarp();

    for (int count = 0, n_iter = N / warp_group_size; count < n_iter; count += 1) {
        x = *px;
        const std::array<nv_bfloat162, 2> *npa = pa;
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            a[i] = *npa;
            npa += stride / 2;
        }
        pa += warpSize;
        px += warpSize;

        auto v0 = __bfloat1622float2(x[0]);
        auto v1 = __bfloat1622float2(x[1]);
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            auto u0 = __bfloat1622float2(a[i][0]);
            auto u1 = __bfloat1622float2(a[i][1]);
            y[i] += (u0.x * v0.x + u0.y * v0.y) + (u1.x * v1.x + u1.y * v1.y);
        }
    }

    // warp reduce on y
    __syncwarp();
    #pragma unroll
    for (int i = 0; i < OP_PER_LANE; i++) {
        #pragma unroll
        for (int j = warpSize / 2; j > 0; j /= 2) {
            y[i] += __shfl_down_sync(0xFFFFFFFF, y[i], j);
        }
    }

    // __syncthreads();
    __syncwarp();

    if (lane_id == 0) {
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            Y[(warp_group_id * OP_PER_LANE - boundry_offset) + i] = __float2bfloat16(y[i]);
        }
    }
}


void gemv_bf16(
    const torch::Tensor &A,
    const torch::Tensor &X,
    torch::Tensor &Y
) {
    int M = A.size(0);
    int N = A.size(1);

    int num_warps_per_block = 2;
    auto block_size = dim3(32, num_warps_per_block, 1);
    int grid_size = ceil_div(M, OP_PER_LANE * num_warps_per_block);

    auto stream = c10::cuda::getCurrentCUDAStream(A.device().index()).stream();

    gemv_bf16_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const nv_bfloat162*>(A.const_data_ptr()),
        static_cast<const nv_bfloat162*>(X.const_data_ptr()),
        static_cast<nv_bfloat16*>(Y.mutable_data_ptr()),
        M, N
    );
}


struct LUT {
    uint8_t LUT1[256];
    uint8_t LUT2[256];
    uint8_t LUT3[256];
    uint8_t LUT4[256];
    uint8_t code_lengths[256];
};


struct decoder{
    union {
        uint64_t data;
        uchar4 v;
    } state{0};
    uint8_t remaining_bits = 0;

    __device__ __inline__ uint8_t decode_symbol(
        const uint32_t* &pae, int warp_group_size,
        const uint8_t* LUT1, const uint8_t* LUT2, const uint8_t* LUT3, const uint8_t* LUT4,
        const uint8_t* code_lengths
    ) {
        uint8_t symbol;

        if (remaining_bits < 32) {
            state.data |= uint64_t(*pae) << remaining_bits;
            pae += warp_group_size;
            remaining_bits += 32;
        }
        
        if ((symbol = LUT1[state.v.x]) != 255);
        else if ((symbol = LUT2[state.v.y]) != 255);
        else if ((symbol = LUT3[state.v.z]) != 255);
        else if ((symbol = LUT4[state.v.w]) != 255);
        // else assert(0);
        auto bitoffset = code_lengths[symbol];
        state.data >>= bitoffset;
        remaining_bits -= bitoffset;

        return symbol;
    }

    __device__ __inline__ uint8_t decode_symbol2(
        const uint32_t* &pae, int warp_group_size, const LUT *lut
    ) {
        uint8_t symbol;

        if (remaining_bits < 32) {
            state.data |= uint64_t(*pae) << remaining_bits;
            pae += warp_group_size;
            remaining_bits += 32;
        }
        
        if ((symbol = lut->LUT1[state.v.x]) != 255);
        else if ((symbol = lut->LUT2[state.v.y]) != 255);
        else if ((symbol = lut->LUT3[state.v.z]) != 255);
        else if ((symbol = lut->LUT4[state.v.w]) != 255);
        // else assert(0);
        auto bitoffset = lut->code_lengths[symbol];
        state.data >>= bitoffset;
        remaining_bits -= bitoffset;

        return symbol;
    }
};


template <int batch_size>
__global__ void
gemv_bf16_huffman_kernel(
    const uchar2* A_rem, const uint32_t* A_exp, const nv_bfloat162* X, nv_bfloat16* Y,
    const uint32_t* offsets,
    const uint8_t* LUT1, const uint8_t* LUT2, const uint8_t* LUT3, const uint8_t* LUT4,
    const uint8_t* code_lengths,
    int M, int N, int split_k
) {
    __shared__ LUT sh_LUT;

    ((uint64_t*)sh_LUT.LUT1)[threadIdx.x] = ((const uint64_t*)LUT1)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT2)[threadIdx.x] = ((const uint64_t*)LUT2)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT3)[threadIdx.x] = ((const uint64_t*)LUT3)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT4)[threadIdx.x] = ((const uint64_t*)LUT4)[threadIdx.x];
    ((uint64_t*)sh_LUT.code_lengths)[threadIdx.x] = ((const uint64_t*)code_lengths)[threadIdx.x];

    __syncthreads();

    int thread_id = ((blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) * 2;

    int warp_group_size = warpSize * 2;

    int warp_group_id = thread_id / warp_group_size;
    int lane_id = thread_id % warp_group_size;

    if (warp_group_id * OP_PER_LANE > M) {
        return; // no work to do
    }

    float y[batch_size][OP_PER_LANE] = {};

    for (int k = 0; k < split_k; k++) {
        int stride = N / 2;

        const uchar4 *par = (const uchar4 *)&A_rem[(warp_group_id * OP_PER_LANE) * stride + lane_id];
        const uint32_t *pae = &A_exp[offsets[warp_group_id] + lane_id];
        const std::array<nv_bfloat162, 2> *px = (const std::array<nv_bfloat162, 2> *)&X[lane_id];

        const uint32_t *pae2 = &A_exp[offsets[warp_group_id] + lane_id + 1];

        std::array<nv_bfloat162, 2> x[batch_size];
        uchar4 ar[OP_PER_LANE];
        uchar4 ae[OP_PER_LANE];

        decoder dec;
        decoder dec2;

        __syncwarp();

        for (int count = 0, n_iter = N / (2 * warp_group_size); count < n_iter; count += 1) {
            #pragma unroll
            for (int i = 0; i < batch_size; i++) {
                x[i] = px[i * N / (sizeof(X[0]) / sizeof(nv_bfloat16))];
            }
            const uchar4 *npar = par;
            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                ar[i] = *npar;
                npar += stride / 2;
            }
            par += warpSize;
            px += warpSize;

            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                ae[i].x = dec.decode_symbol2(pae, warp_group_size, &sh_LUT);
                ae[i].z = dec2.decode_symbol2(pae2, warp_group_size, &sh_LUT);
                ae[i].y = dec.decode_symbol2(pae, warp_group_size, &sh_LUT);
                ae[i].w = dec2.decode_symbol2(pae2, warp_group_size, &sh_LUT);
            }

            // __syncwarp();

            float2 v0[batch_size], v1[batch_size];
            #pragma unroll
            for (int i = 0; i < batch_size; i++) {
                v0[i] = __bfloat1622float2(x[i][0]);
                v1[i] = __bfloat1622float2(x[i][1]);
            }

            // auto v0 = __bfloat1622float2(x[0]);
            // auto v1 = __bfloat1622float2(x[1]);

            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                uint32_t rem0 = (uint32_t(ar[i].y) << 16) | ar[i].x;
                uint32_t rem1 = (uint32_t(ar[i].w) << 16) | ar[i].z;
                uint32_t exp0 = (uint32_t(ae[i].y) << 16) | ae[i].x;
                uint32_t exp1 = (uint32_t(ae[i].w) << 16) | ae[i].z;
                union {
                    uint32_t _bits;
                    nv_bfloat162 u;
                } bf160{((rem0 << 8) & 0x80008000) | (rem0 & 0x007F007F) | (exp0 << 7)};
                union {
                    uint32_t _bits;
                    nv_bfloat162 u;
                } bf161{((rem1 << 8) & 0x80008000) | (rem1 & 0x007F007F) | (exp1 << 7)};
                auto u0 = __bfloat1622float2(bf160.u);
                auto u1 = __bfloat1622float2(bf161.u);
                #pragma unroll
                for (int j = 0; j < batch_size; j++) {
                    y[j][i] += (u0.x * v0[j].x + u0.y * v0[j].y) + (u1.x * v1[j].x + u1.y * v1[j].y);
                }
            }
        }
        
        {
            // handle split k
            int num_warp_groups = blockDim.y * gridDim.x;
            int offsets_stride = num_warp_groups;
            // printf("%d\n", offsets_stride);

            // N /= split_k;
            A_rem += M * N / sizeof(A_rem[0]);
            X += batch_size * N / (sizeof(X[0]) / sizeof(nv_bfloat16));
            offsets += offsets_stride;
        }
    }

    // warp reduce on y
    __syncwarp();
    #pragma unroll
    for (int b = 0; b < batch_size; b++) {
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            #pragma unroll
            for (int j = warpSize / 2; j > 0; j /= 2) {
                y[b][i] += __shfl_down_sync(0xFFFFFFFF, y[b][i], j);
            }
        }
    }

    // __syncthreads();
    __syncwarp();

    if (lane_id == 0) {
        #pragma unroll
        for (int b = 0; b < batch_size; b++) {
            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                Y[(warp_group_id * OP_PER_LANE) + i] = __float2bfloat16(y[b][i]);
                // atomicAdd(&Y[(warp_group_id * OP_PER_LANE) + i], __float2bfloat16(y[i]));
            }
            Y += M;
        }
    }
}


void gemv_bf16_huffman(
    const torch::Tensor &A_rem,
    const torch::Tensor &A_exp,
    const torch::Tensor &X,
    torch::Tensor &Y,
    const torch::Tensor &offsets,
    const torch::Tensor &LUT1,
    const torch::Tensor &LUT2,
    const torch::Tensor &LUT3,
    const torch::Tensor &LUT4,
    const torch::Tensor &code_lengths
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    int num_warps_per_block = 4; // TODO: If 3 will crash randomly
    auto block_size = dim3(32, num_warps_per_block, 1);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = c10::cuda::getCurrentCUDAStream(A_rem.device().index()).stream();

    gemv_bf16_huffman_kernel<1><<<grid_size, block_size, 0, stream>>>(
        static_cast<const uchar2*>(A_rem.const_data_ptr()),
        static_cast<const uint32_t*>(A_exp.const_data_ptr()),
        static_cast<const nv_bfloat162*>(X.const_data_ptr()),
        static_cast<nv_bfloat16*>(Y.mutable_data_ptr()),
        static_cast<const uint32_t*>(offsets.const_data_ptr()),
        static_cast<const uint8_t*>(LUT1.const_data_ptr()),
        static_cast<const uint8_t*>(LUT2.const_data_ptr()),
        static_cast<const uint8_t*>(LUT3.const_data_ptr()),
        static_cast<const uint8_t*>(LUT4.const_data_ptr()),
        static_cast<const uint8_t*>(code_lengths.const_data_ptr()),
        M, N, split_k
    );
}


__global__ void huffman_decode_kernel(
    const uchar2* A_rem, const uint32_t* A_exp, nv_bfloat162* Y,
    const uint32_t* offsets,
    const uint8_t* LUT1, const uint8_t* LUT2, const uint8_t* LUT3, const uint8_t* LUT4,
    const uint8_t* code_lengths,
    int M, int N, int split_k
) {
    __shared__ LUT sh_LUT;

    ((uint64_t*)sh_LUT.LUT1)[threadIdx.x] = ((const uint64_t*)LUT1)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT2)[threadIdx.x] = ((const uint64_t*)LUT2)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT3)[threadIdx.x] = ((const uint64_t*)LUT3)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT4)[threadIdx.x] = ((const uint64_t*)LUT4)[threadIdx.x];
    ((uint64_t*)sh_LUT.code_lengths)[threadIdx.x] = ((const uint64_t*)code_lengths)[threadIdx.x];

    __syncthreads();

    int thread_id = ((blockIdx.x * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) * 2;

    int warp_group_size = warpSize * 2;

    int warp_group_id = thread_id / warp_group_size;
    int lane_id = thread_id % warp_group_size;

    if (warp_group_id * OP_PER_LANE > M) {
        return; // no work to do
    }

    for (int k = 0; k < split_k; k++) {
        int stride = N / 2;

        const uchar4 *par = (const uchar4 *)&A_rem[(warp_group_id * OP_PER_LANE) * stride + lane_id];
        const uint32_t *pae = &A_exp[offsets[warp_group_id] + lane_id];
        const uint32_t *pae2 = &A_exp[offsets[warp_group_id] + lane_id + 1];

        uchar4 ar[OP_PER_LANE];
        uchar4 ae[OP_PER_LANE];

        decoder dec;
        decoder dec2;

        __syncwarp();

        for (int count = 0, n_iter = N / (2 * warp_group_size); count < n_iter; count += 1) {
            const uchar4 *npar = par;
            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                ar[i] = *npar;
                npar += stride / 2;
            }
            par += warpSize;

            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                ae[i].x = dec.decode_symbol2(pae, warp_group_size, &sh_LUT);
                ae[i].z = dec2.decode_symbol2(pae2, warp_group_size, &sh_LUT);
                ae[i].y = dec.decode_symbol2(pae, warp_group_size, &sh_LUT);
                ae[i].w = dec2.decode_symbol2(pae2, warp_group_size, &sh_LUT);
            }

            // __syncwarp();

            #pragma unroll
            for (int i = 0; i < OP_PER_LANE; i++) {
                uint32_t rem0 = (uint32_t(ar[i].y) << 16) | ar[i].x;
                uint32_t rem1 = (uint32_t(ar[i].w) << 16) | ar[i].z;
                uint32_t exp0 = (uint32_t(ae[i].y) << 16) | ae[i].x;
                uint32_t exp1 = (uint32_t(ae[i].w) << 16) | ae[i].z;
                union {
                    uint32_t _bits;
                    nv_bfloat162 u;
                } bf160{((rem0 << 8) & 0x80008000) | (rem0 & 0x007F007F) | (exp0 << 7)};
                union {
                    uint32_t _bits;
                    nv_bfloat162 u;
                } bf161{((rem1 << 8) & 0x80008000) | (rem1 & 0x007F007F) | (exp1 << 7)};
                Y[(warp_group_id * OP_PER_LANE + i) * N / 2 + count * warp_group_size + lane_id] = bf160.u;
                Y[(warp_group_id * OP_PER_LANE + i) * N / 2 + count * warp_group_size + lane_id + 1] = bf161.u;
            }
        }
        
        {
            // handle split k
            int num_warp_groups = blockDim.y * gridDim.x;
            int offsets_stride = num_warp_groups;
            // printf("%d\n", offsets_stride);

            // N /= split_k;
            A_rem += M * N / sizeof(A_rem[0]);
            Y += M * N / (sizeof(Y[0]) / sizeof(nv_bfloat16));
            offsets += offsets_stride;
        }
    }
}


void huffman_decode(
    const torch::Tensor &A_rem,
    const torch::Tensor &A_exp,
    torch::Tensor &Y,
    const torch::Tensor &offsets,
    const torch::Tensor &LUT1,
    const torch::Tensor &LUT2,
    const torch::Tensor &LUT3,
    const torch::Tensor &LUT4,
    const torch::Tensor &code_lengths
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    int num_warps_per_block = 4; // TODO: If 3 will crash randomly
    auto block_size = dim3(32, num_warps_per_block, 1);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = c10::cuda::getCurrentCUDAStream(A_rem.device().index()).stream();

    huffman_decode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uchar2*>(A_rem.const_data_ptr()),
        static_cast<const uint32_t*>(A_exp.const_data_ptr()),
        static_cast<nv_bfloat162*>(Y.mutable_data_ptr()),
        static_cast<const uint32_t*>(offsets.const_data_ptr()),
        static_cast<const uint8_t*>(LUT1.const_data_ptr()),
        static_cast<const uint8_t*>(LUT2.const_data_ptr()),
        static_cast<const uint8_t*>(LUT3.const_data_ptr()),
        static_cast<const uint8_t*>(LUT4.const_data_ptr()),
        static_cast<const uint8_t*>(code_lengths.const_data_ptr()),
        M, N, split_k
    );
}


__global__ void huffman_encode_kernel(
    const uint8_t *data,
    uint32_t data_length,
    int num_data,
    const char* LUT,
    char *output,
    uint32_t output_lengths[]
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= num_data) return;

    uint32_t output_count = 0;
    for (int i = 0; i < data_length; i++) {
        const char *p = &LUT[data[thread_id * data_length + i] * 32];
        for (char ch = *p++, count = 0; ch != '\0' && count < 32; count++, ch = *p++) {
            output[thread_id * data_length * 32 + output_count] = ch;
            output_count++;
        }
    }
    output_lengths[thread_id] = output_count;
}


void huffman_encode(
    const torch::Tensor &data,
    const torch::Tensor &LUT,
    torch::Tensor &output,
    torch::Tensor &output_lengths
) { 
    int num_data = data.size(0);
    int data_lengths = data.size(1);
    int block_size = 32;
    int grid_size = ceil_div(num_data, block_size);
    auto stream = c10::cuda::getCurrentCUDAStream(data.device().index()).stream();
    huffman_encode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uint8_t*>(data.const_data_ptr()),
        data_lengths,
        num_data,
        static_cast<const char*>(LUT.const_data_ptr()),
        static_cast<char*>(output.mutable_data_ptr()),
        static_cast<uint32_t*>(output_lengths.mutable_data_ptr())
    );
}

TORCH_LIBRARY_IMPL(bf16_huffman_infer, CUDA, m) {
    m.impl("gemv_bf16", &gemv_bf16);
    m.impl("gemv_bf16_huffman", &gemv_bf16_huffman);
    m.impl("huffman_encode", &huffman_encode);
    m.impl("huffman_decode", &huffman_decode);
}

}
