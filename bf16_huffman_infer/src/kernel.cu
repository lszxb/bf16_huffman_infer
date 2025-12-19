#define USE_CUDA

#include <torch/csrc/stable/library.h>
// #include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <array>


#define REP_1_8(x, y, ...) \
    { constexpr int x = 1; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 2; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 3; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 4; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 5; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 6; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 7; if (y == x) {__VA_ARGS__;} } \
    { constexpr int x = 8; if (y == x) {__VA_ARGS__;} }


#define OP_PER_LANE 1
#define MAX_WARP_BLOCK_RATIO 4
#define MAX_SPLIT_K 32
#define ANS_PRECISION 10

namespace bf16_huffman_infer {


cudaStream_t get_tensor_stream(const torch::stable::Tensor &t) {
    // The following stable abi now will crash while capturing cuda graph, so use the low level once.
    // return (cudaStream_t)torch::stable::accelerator::getCurrentStream(t.get_device_index()).id();
    
    // This is from https://github.com/facebookresearch/xformers/blob/0b76a47e097ff94b33824ef1df9511278f80da3f/xformers/csrc/pt_stable_utils.h#L35
    void* ret;
    TORCH_ERROR_CODE_CHECK(aoti_torch_get_current_cuda_stream(t.get_device_index(), &ret));
    return static_cast<cudaStream_t>(ret);
}


static int ceil_div(int a, int b) {
    return (a + b - 1) / b;
}


template <int batch_size, typename decoder, typename LUT>
__device__ __inline__ void
gemv_bf16_general_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, const nv_bfloat162* X, nv_bfloat16* Y,
    const uint32_t* offsets,
    const LUT* decoder_LUT,
    int M, int N, int split_k
) {
    __shared__ struct {
        float y[MAX_WARP_BLOCK_RATIO][batch_size][OP_PER_LANE][MAX_SPLIT_K];
        int count[MAX_WARP_BLOCK_RATIO];
    } tmp;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        tmp.count[threadIdx.z] = 0;
    }
    assert(blockDim.z <= MAX_WARP_BLOCK_RATIO);
    assert(split_k <= MAX_SPLIT_K);

    __syncthreads();

    assert(blockDim.x == warpSize);

    int warp_group_id = blockIdx.x * blockDim.z + threadIdx.z;
    int lane_id = threadIdx.x;
    int thread_id = warp_group_id * blockDim.x + threadIdx.x;

    if (warp_group_id * OP_PER_LANE >= M) {
        return; // no work to do
    }

    float y[batch_size][OP_PER_LANE] = {};

    int k = threadIdx.y;
    // int k = warp_group_id / (M / OP_PER_LANE);
    // warp_group_id %= (M / OP_PER_LANE);

    A_rem += M * N / sizeof(A_rem[0]) * k;
    X += N / (sizeof(X[0]) / sizeof(nv_bfloat16)) * k;
    offsets += M * k;

    int stride = N / 4;

    // const vec<nv_bfloat162, 2> *px = &X[lane_id];
    const nv_bfloat162 *px = &X[lane_id];
    const uchar4 *par = &A_rem[(warp_group_id * OP_PER_LANE) * stride + lane_id];

    const uint32_t *pae0 = &A_exp[offsets[warp_group_id] + lane_id + 0];
    const uint32_t *pae1 = &A_exp[offsets[warp_group_id] + lane_id + warpSize];

    // vec<nv_bfloat162, 2> x[batch_size];
    nv_bfloat162 x[batch_size][2];
    uchar4 ar[OP_PER_LANE];
    uchar4 ae[OP_PER_LANE];

    decoder dec0;
    decoder dec1;

    __syncwarp();

    for (int count = 0, n_iter = N / (4 * warpSize); count < n_iter; count += 1) {
        #pragma unroll
        for (int i = 0; i < batch_size; i++) {
            // NOTE: it will not work as expected: vector load 64bit, if using array<nv_bfloat162,2>
            // instead, it load 2 32bits load, with interleaved layout, which is much slower
            // x[i] = px[i * (split_k * N / (sizeof(px[0]) / sizeof(nv_bfloat16)))];
            x[i][0] = px[i * (split_k * N / (sizeof(px[0]) / sizeof(nv_bfloat16))) + 0];
            x[i][1] = px[i * (split_k * N / (sizeof(px[0]) / sizeof(nv_bfloat16))) + warpSize];
        }
        const uchar4 *npar = par;
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            ar[i] = *npar;
            npar += stride;
        }
        par += warpSize;
        px += warpSize * 2;

        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            ae[i].x = dec0.decode_symbol2(pae0, warpSize * 2, decoder_LUT);
            ae[i].z = dec1.decode_symbol2(pae1, warpSize * 2, decoder_LUT);
            ae[i].y = dec0.decode_symbol2(pae0, warpSize * 2, decoder_LUT);
            ae[i].w = dec1.decode_symbol2(pae1, warpSize * 2, decoder_LUT);
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
                // Y[(warp_group_id * OP_PER_LANE) + i] = __float2bfloat16(y[b][i]);
                // atomicAdd(&Y[(warp_group_id * OP_PER_LANE) + i], __float2bfloat16(y[b][i]));
                // atomicAdd(&Y[(warp_group_id * OP_PER_LANE) + i], y[b][i]);
                tmp.y[threadIdx.z][b][i][k] = y[b][i];
            }
            // Y += M;
        }
        // Y -= M * batch_size; // reset Y pointer to the start of the batch

        int res = atomicAdd_block(&tmp.count[threadIdx.z], 1);
        if (res == split_k - 1) {
            // last thread in the block, write back the results
            #pragma unroll
            for (int b = 0; b < batch_size; b++) {
                #pragma unroll
                for (int i = 0; i < OP_PER_LANE; i++) {
                    float y = 0.0;
                    for (int j = 0; j < split_k; j++) {
                        y += tmp.y[threadIdx.z][b][i][j];
                    }
                    Y[(warp_group_id * OP_PER_LANE) + i] = __float2bfloat16(y);
                }
                Y += M;
            }
            Y -= M * batch_size; // reset Y pointer to the start of the batch
        }
    }
}


template <typename decoder, typename LUT>
__device__ __inline__ void
decode_general_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, nv_bfloat162* Y,
    const uint32_t* offsets,
    const LUT* decoder_LUT,
    int M, int N, int split_k
) {
    assert(blockDim.z <= MAX_WARP_BLOCK_RATIO);
    assert(split_k <= MAX_SPLIT_K);

    __syncthreads();

    assert(blockDim.x == warpSize);

    int warp_group_id = blockIdx.x * blockDim.z + threadIdx.z;
    int lane_id = threadIdx.x;
    int thread_id = warp_group_id * blockDim.x + threadIdx.x;

    if (warp_group_id * OP_PER_LANE >= M) {
        return; // no work to do
    }

    Y += warp_group_id * OP_PER_LANE * (N * split_k) / (sizeof(Y[0]) / sizeof(nv_bfloat16));

    int k = threadIdx.y;
    // int k = warp_group_id / (M / OP_PER_LANE);
    // warp_group_id %= (M / OP_PER_LANE);

    A_rem += M * N / sizeof(A_rem[0]) * k;
    Y += N / (sizeof(Y[0]) / sizeof(nv_bfloat16)) * k;
    offsets += M * k;

    int stride = N / 4;

    nv_bfloat162 *py = &Y[lane_id];
    const uchar4 *par = &A_rem[(warp_group_id * OP_PER_LANE) * stride + lane_id];

    const uint32_t *pae0 = &A_exp[offsets[warp_group_id] + lane_id + 0];
    const uint32_t *pae1 = &A_exp[offsets[warp_group_id] + lane_id + warpSize];

    uchar4 ar[OP_PER_LANE];
    uchar4 ae[OP_PER_LANE];

    decoder dec0;
    decoder dec1;

    __syncwarp();

    for (int count = 0, n_iter = N / (4 * warpSize); count < n_iter; count += 1) {
        const uchar4 *npar = par;
        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            ar[i] = *npar;
            npar += stride;
        }
        par += warpSize;

        #pragma unroll
        for (int i = 0; i < OP_PER_LANE; i++) {
            ae[i].x = dec0.decode_symbol2(pae0, warpSize * 2, decoder_LUT);
            ae[i].z = dec1.decode_symbol2(pae1, warpSize * 2, decoder_LUT);
            ae[i].y = dec0.decode_symbol2(pae0, warpSize * 2, decoder_LUT);
            ae[i].w = dec1.decode_symbol2(pae1, warpSize * 2, decoder_LUT);
        }

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
            py[i * (split_k * N / (sizeof(py[0]) / sizeof(nv_bfloat16))) + 0] = bf160.u;
            py[i * (split_k * N / (sizeof(py[0]) / sizeof(nv_bfloat16))) + warpSize] = bf161.u;
        }
        py += warpSize * 2;
    }
}


struct huffman_LUT {
    uint8_t LUT1[256];
    uint8_t LUT2[256];
    uint8_t LUT3[256];
    uint8_t LUT4[256];
    uint8_t code_lengths[256];
};


struct huffman_decoder{
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
        const uint32_t* &pae, int warp_group_size, const huffman_LUT *lut
    ) {
        uint8_t symbol;

        if (remaining_bits < 32) {
            // int num = __popc(__activemask());
            // if (blockIdx.x == 0 && threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
            //     printf("active cnt: %d\n", num);
            // }
            // TODO: *pae is interleaved load, fix it
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


template <typename T, int width>
union vec {
    template <int width> struct width_to_vector_type {};
    template <> struct width_to_vector_type<1> { using type = uint1; };
    template <> struct width_to_vector_type<2> { using type = uint2; };
    template <> struct width_to_vector_type<4> { using type = uint4; };


    using vector_type = typename width_to_vector_type<width>::type;
    vector_type data;
    T value[width];

    __device__ __inline__ vec<T, width>& operator=(const vec<T, width>& other) {
        data = other.data;
        return *this;
    }

    __device__ __inline__ vec<T, width>& operator=(vec<T, width>&& other) {
        data = other.data;
        return *this;
    }

    template <typename I>
    __device__ __inline__ T operator[](I index) {
        return value[index];
    }
};


template <int batch_size>
__global__ void
gemv_bf16_huffman_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, const nv_bfloat162* X, nv_bfloat16* Y,
    const uint32_t* offsets,
    const uint8_t* LUT1, const uint8_t* LUT2, const uint8_t* LUT3, const uint8_t* LUT4,
    const uint8_t* code_lengths,
    int M, int N, int split_k
) {
    __shared__ huffman_LUT sh_LUT;

    ((uint64_t*)sh_LUT.LUT1)[threadIdx.x] = ((const uint64_t*)LUT1)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT2)[threadIdx.x] = ((const uint64_t*)LUT2)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT3)[threadIdx.x] = ((const uint64_t*)LUT3)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT4)[threadIdx.x] = ((const uint64_t*)LUT4)[threadIdx.x];
    ((uint64_t*)sh_LUT.code_lengths)[threadIdx.x] = ((const uint64_t*)code_lengths)[threadIdx.x];

    gemv_bf16_general_kernel<batch_size, huffman_decoder, huffman_LUT>(
        A_rem, A_exp, X, Y,
        offsets,
        &sh_LUT,
        M, N, split_k
    );
}


void gemv_bf16_huffman(
    const torch::stable::Tensor &A_rem,
    const torch::stable::Tensor &A_exp,
    const torch::stable::Tensor &X,
    torch::stable::Tensor &Y,
    const torch::stable::Tensor &offsets,
    const torch::stable::Tensor &LUT1,
    const torch::stable::Tensor &LUT2,
    const torch::stable::Tensor &LUT3,
    const torch::stable::Tensor &LUT4,
    const torch::stable::Tensor &code_lengths
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    cudaDeviceProp attr;
    STD_TORCH_CHECK(cudaGetDeviceProperties(&attr, A_rem.get_device_index()) == cudaSuccess);
    int num_warps_per_block = attr.maxThreadsPerMultiProcessor / 32 / attr.maxBlocksPerMultiProcessor;
    num_warps_per_block = ceil_div(num_warps_per_block, split_k);

    auto block_size = dim3(32, split_k, num_warps_per_block);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = get_tensor_stream(A_rem);

    int batch_size = X.size(0);
    STD_TORCH_CHECK(batch_size <= 8);

    REP_1_8(
        b, batch_size,
        gemv_bf16_huffman_kernel<b><<<grid_size, block_size, 0, stream>>>(
            static_cast<const uchar4*>(A_rem.data_ptr()),
            static_cast<const uint32_t*>(A_exp.data_ptr()),
            static_cast<const nv_bfloat162*>(X.data_ptr()),
            static_cast<nv_bfloat16*>(Y.data_ptr()),
            static_cast<const uint32_t*>(offsets.data_ptr()),
            static_cast<const uint8_t*>(LUT1.data_ptr()),
            static_cast<const uint8_t*>(LUT2.data_ptr()),
            static_cast<const uint8_t*>(LUT3.data_ptr()),
            static_cast<const uint8_t*>(LUT4.data_ptr()),
            static_cast<const uint8_t*>(code_lengths.data_ptr()),
            M, N, split_k
        )
    );
}


void boxed_gemv_bf16_huffman(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto Y = to<torch::stable::Tensor>(stack[3]);
    gemv_bf16_huffman(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        to<torch::stable::Tensor>(stack[2]),
        Y,
        to<torch::stable::Tensor>(stack[4]),
        to<torch::stable::Tensor>(stack[5]),
        to<torch::stable::Tensor>(stack[6]),
        to<torch::stable::Tensor>(stack[7]),
        to<torch::stable::Tensor>(stack[8]),
        to<torch::stable::Tensor>(stack[9])
    );
}


__global__ void huffman_decode_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, nv_bfloat162* Y,
    const uint32_t* offsets,
    const uint8_t* LUT1, const uint8_t* LUT2, const uint8_t* LUT3, const uint8_t* LUT4,
    const uint8_t* code_lengths,
    int M, int N, int split_k
) {
    __shared__ huffman_LUT sh_LUT;

    ((uint64_t*)sh_LUT.LUT1)[threadIdx.x] = ((const uint64_t*)LUT1)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT2)[threadIdx.x] = ((const uint64_t*)LUT2)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT3)[threadIdx.x] = ((const uint64_t*)LUT3)[threadIdx.x];
    ((uint64_t*)sh_LUT.LUT4)[threadIdx.x] = ((const uint64_t*)LUT4)[threadIdx.x];
    ((uint64_t*)sh_LUT.code_lengths)[threadIdx.x] = ((const uint64_t*)code_lengths)[threadIdx.x];

    decode_general_kernel<huffman_decoder, huffman_LUT>(
        A_rem, A_exp, Y,
        offsets,
        &sh_LUT,
        M, N, split_k
    );
}


void huffman_decode(
    const torch::stable::Tensor &A_rem,
    const torch::stable::Tensor &A_exp,
    torch::stable::Tensor &Y,
    const torch::stable::Tensor &offsets,
    const torch::stable::Tensor &LUT1,
    const torch::stable::Tensor &LUT2,
    const torch::stable::Tensor &LUT3,
    const torch::stable::Tensor &LUT4,
    const torch::stable::Tensor &code_lengths
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    cudaDeviceProp attr;
    STD_TORCH_CHECK(cudaGetDeviceProperties(&attr, A_rem.get_device_index()) == cudaSuccess);
    int num_warps_per_block = attr.maxThreadsPerMultiProcessor / 32 / attr.maxBlocksPerMultiProcessor;
    num_warps_per_block = ceil_div(num_warps_per_block, split_k);

    auto block_size = dim3(32, split_k, num_warps_per_block);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = get_tensor_stream(A_rem);

    huffman_decode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uchar4*>(A_rem.data_ptr()),
        static_cast<const uint32_t*>(A_exp.data_ptr()),
        static_cast<nv_bfloat162*>(Y.data_ptr()),
        static_cast<const uint32_t*>(offsets.data_ptr()),
        static_cast<const uint8_t*>(LUT1.data_ptr()),
        static_cast<const uint8_t*>(LUT2.data_ptr()),
        static_cast<const uint8_t*>(LUT3.data_ptr()),
        static_cast<const uint8_t*>(LUT4.data_ptr()),
        static_cast<const uint8_t*>(code_lengths.data_ptr()),
        M, N, split_k
    );
}


void boxed_huffman_decode(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto Y = to<torch::stable::Tensor>(stack[2]);
    huffman_decode(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        Y,
        to<torch::stable::Tensor>(stack[3]),
        to<torch::stable::Tensor>(stack[4]),
        to<torch::stable::Tensor>(stack[5]),
        to<torch::stable::Tensor>(stack[6]),
        to<torch::stable::Tensor>(stack[7]),
        to<torch::stable::Tensor>(stack[8])
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
    const torch::stable::Tensor &data,
    const torch::stable::Tensor &LUT,
    torch::stable::Tensor &output,
    torch::stable::Tensor &output_lengths
) {
    int num_data = data.size(0);
    int data_lengths = data.size(1);
    int block_size = 32;
    int grid_size = ceil_div(num_data, block_size);
    auto stream = get_tensor_stream(data);
    huffman_encode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uint8_t*>(data.data_ptr()),
        data_lengths,
        num_data,
        static_cast<const char*>(LUT.data_ptr()),
        static_cast<char*>(output.data_ptr()),
        static_cast<uint32_t*>(output_lengths.data_ptr())
    );
}


void boxed_huffman_encode(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto output = to<torch::stable::Tensor>(stack[2]);
    auto output_lengths = to<torch::stable::Tensor>(stack[3]);
    huffman_encode(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        output,
        output_lengths
    );
}


struct ans_LUT {
    uint32_t rem: 12, freq: 12, sym: 8;
};


struct ans_decoder{
    uint64_t state = 0xffffffffffffffff;

    __device__ __inline__ uint8_t decode_symbol2(
        const uint32_t* &pae, int warp_group_size, const ans_LUT *lut
    ) {
        if (state == 0xffffffffffffffff) {
            state = (*pae);
            pae += warp_group_size;
        }

        // __syncwarp();

        if (state < (1 << ANS_PRECISION)) {
            // int num = __popc(__activemask());
            // if (blockIdx.x == 0 && threadIdx.z == 0 && threadIdx.y == 0 && threadIdx.x == 0) {
            //     printf("active cnt: %d\n", num);
            // }
            state = (state << 32) | (*pae);
            pae += warp_group_size;
        }

        uint32_t slot = state & ((1 << ANS_PRECISION) - 1);
        auto res = lut[slot];
        state = res.freq * (state >> ANS_PRECISION) + res.rem;

        return res.sym;
    }
};


template <int batch_size>
__global__ void
gemv_bf16_ans_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, const nv_bfloat162* X, nv_bfloat16* Y,
    const uint32_t* offsets,
    const ans_LUT* LUT,
    int M, int N, int split_k
) {
    static_assert(sizeof(ans_LUT) == 4, "ans_LUT size must be 4 bytes");
    static_assert(ANS_PRECISION <= 12, "ANS_PRECISION must be less than or equal to 12");
    __shared__ ans_LUT sh_LUT[1 << ANS_PRECISION];

    int thread_idx = threadIdx.z * blockDim.y + threadIdx.y;
    thread_idx = thread_idx * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int i = thread_idx; i < (1 << ANS_PRECISION); i += block_size) {
        sh_LUT[i] = LUT[i];
    }    

    gemv_bf16_general_kernel<batch_size, ans_decoder, ans_LUT>(
        A_rem, A_exp, X, Y,
        offsets,
        sh_LUT,
        M, N, split_k
    );
}


void gemv_bf16_ans(
    const torch::stable::Tensor &A_rem,
    const torch::stable::Tensor &A_exp,
    const torch::stable::Tensor &X,
    torch::stable::Tensor &Y,
    const torch::stable::Tensor &offsets,
    const torch::stable::Tensor &LUT
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    cudaDeviceProp attr;
    STD_TORCH_CHECK(cudaGetDeviceProperties(&attr, A_rem.get_device_index()) == cudaSuccess);
    int num_warps_per_block = attr.maxThreadsPerMultiProcessor / 32 / attr.maxBlocksPerMultiProcessor;
    num_warps_per_block = ceil_div(num_warps_per_block, split_k);

    auto block_size = dim3(32, split_k, num_warps_per_block);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = get_tensor_stream(A_rem);

    int batch_size = X.size(0);
    STD_TORCH_CHECK(batch_size <= 8);

    REP_1_8(
        b, batch_size,
        gemv_bf16_ans_kernel<b><<<grid_size, block_size, 0, stream>>>(
            static_cast<const uchar4*>(A_rem.data_ptr()),
            static_cast<const uint32_t*>(A_exp.data_ptr()),
            static_cast<const nv_bfloat162*>(X.data_ptr()),
            static_cast<nv_bfloat16*>(Y.data_ptr()),
            static_cast<const uint32_t*>(offsets.data_ptr()),
            static_cast<const ans_LUT*>(LUT.data_ptr()),
            M, N, split_k
        )
    );
}


void boxed_gemv_bf16_ans(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto Y = to<torch::stable::Tensor>(stack[3]);
    gemv_bf16_ans(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        to<torch::stable::Tensor>(stack[2]),
        Y,
        to<torch::stable::Tensor>(stack[4]),
        to<torch::stable::Tensor>(stack[5])
    );
}


__global__ void ans_encode_kernel(
    const uint8_t *data,
    uint32_t data_length,
    int num_data,
    const uint32_t* freq,
    const uint32_t* cum,
    uint32_t *output,
    uint32_t output_lengths[]
) {
    int thread_id = threadIdx.x + blockIdx.x * blockDim.x;
    if (thread_id >= num_data) return;
    
    uint64_t state = 1 << ANS_PRECISION;
    uint32_t output_count = 0;
    for (int i = 0; i < data_length; i++) {
        auto sym = data[thread_id * data_length + data_length - i - 1];
        auto f = freq[sym];
        auto start = cum[sym];

        while ((state >> 32) >= f) {
            output[thread_id * data_length * 8 + output_count] = (uint32_t)state;
            output_count++;
            state >>= 32;
        }

        auto quotient = state / f;
        auto remainder = state % f;

        state = (quotient << ANS_PRECISION) + remainder + start;
    }

    output[thread_id * data_length * 8 + output_count] = (uint32_t)state;
    output_count++;
    state >>= 32;

    output[thread_id * data_length * 8 + output_count] = (uint32_t)state;
    output_count++;
    state >>= 32;

    for (int i = 0; i < output_count / 2; i++) {
        auto tmp = output[thread_id * data_length * 8 + i];
        output[thread_id * data_length * 8 + i] = output[thread_id * data_length * 8 + output_count - i - 1];
        output[thread_id * data_length * 8 + output_count - i - 1] = tmp;
    }

    output_lengths[thread_id] = output_count;
}


void ans_encode(
    const torch::stable::Tensor &data,
    const torch::stable::Tensor &freq,
    const torch::stable::Tensor &cum,
    torch::stable::Tensor &output,
    torch::stable::Tensor &output_lengths
) {
    int num_data = data.size(0);
    int data_lengths = data.size(1);
    int block_size = 32;
    int grid_size = ceil_div(num_data, block_size);
    auto stream = get_tensor_stream(data);
    ans_encode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uint8_t*>(data.data_ptr()),
        data_lengths,
        num_data,
        static_cast<const uint32_t*>(freq.data_ptr()),
        static_cast<const uint32_t*>(cum.data_ptr()),
        static_cast<uint32_t*>(output.data_ptr()),
        static_cast<uint32_t*>(output_lengths.data_ptr())
    );
}


void boxed_ans_encode(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto output = to<torch::stable::Tensor>(stack[3]);
    auto output_lengths = to<torch::stable::Tensor>(stack[4]);
    ans_encode(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        to<torch::stable::Tensor>(stack[2]),
        output,
        output_lengths
    );
}


__global__ void
ans_decode_kernel(
    const uchar4* A_rem, const uint32_t* A_exp, nv_bfloat162* Y,
    const uint32_t* offsets,
    const ans_LUT* LUT,
    int M, int N, int split_k
) {
    static_assert(sizeof(ans_LUT) == 4, "ans_LUT size must be 4 bytes");
    static_assert(ANS_PRECISION <= 12, "ANS_PRECISION must be less than or equal to 12");
    __shared__ ans_LUT sh_LUT[1 << ANS_PRECISION];

    int thread_idx = threadIdx.z * blockDim.y + threadIdx.y;
    thread_idx = thread_idx * blockDim.x + threadIdx.x;
    int block_size = blockDim.x * blockDim.y * blockDim.z;
    for (int i = thread_idx; i < (1 << ANS_PRECISION); i += block_size) {
        sh_LUT[i] = LUT[i];
    }    

    decode_general_kernel<ans_decoder, ans_LUT>(
        A_rem, A_exp, Y,
        offsets,
        sh_LUT,
        M, N, split_k
    );
}


void ans_decode_kernel(
    const torch::stable::Tensor &A_rem,
    const torch::stable::Tensor &A_exp,
    torch::stable::Tensor &Y,
    const torch::stable::Tensor &offsets,
    const torch::stable::Tensor &LUT
) {
    int split_k = A_rem.size(0);
    int M = A_rem.size(1);
    int N = A_rem.size(2);

    cudaDeviceProp attr;
    STD_TORCH_CHECK(cudaGetDeviceProperties(&attr, A_rem.get_device_index()) == cudaSuccess);
    int num_warps_per_block = attr.maxThreadsPerMultiProcessor / 32 / attr.maxBlocksPerMultiProcessor;
    num_warps_per_block = ceil_div(num_warps_per_block, split_k);

    auto block_size = dim3(32, split_k, num_warps_per_block);
    auto grid_size = dim3(ceil_div(M, OP_PER_LANE * num_warps_per_block), 1, 1);

    auto stream = get_tensor_stream(A_rem);

    ans_decode_kernel<<<grid_size, block_size, 0, stream>>>(
        static_cast<const uchar4*>(A_rem.data_ptr()),
        static_cast<const uint32_t*>(A_exp.data_ptr()),
        static_cast<nv_bfloat162*>(Y.data_ptr()),
        static_cast<const uint32_t*>(offsets.data_ptr()),
        static_cast<const ans_LUT*>(LUT.data_ptr()),
        M, N, split_k
    );
}


void boxed_ans_decode(StableIValue* stack, uint64_t num_args, uint64_t num_outputs) {
    auto Y = to<torch::stable::Tensor>(stack[2]);
    ans_decode_kernel(
        to<torch::stable::Tensor>(stack[0]),
        to<torch::stable::Tensor>(stack[1]),
        Y,
        to<torch::stable::Tensor>(stack[3]),
        to<torch::stable::Tensor>(stack[4])
    );
}


STABLE_TORCH_LIBRARY_IMPL(bf16_huffman_infer, CUDA, m) {
    m.impl("gemv_bf16_huffman", &boxed_gemv_bf16_huffman);
    m.impl("huffman_encode", &boxed_huffman_encode);
    m.impl("huffman_decode", &boxed_huffman_decode);

    m.impl("gemv_bf16_ans", &boxed_gemv_bf16_ans);
    m.impl("ans_encode", &boxed_ans_encode);
    m.impl("ans_decode", &boxed_ans_decode);
}

}
