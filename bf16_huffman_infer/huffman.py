import numpy as np
import heapq
from typing import Dict, List, Tuple, Union
import torch
from collections import defaultdict
from numba import jit

from tqdm import tqdm

class HuffmanNode:
    """Huffman树节点"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol  # 符号（字节值0-255）
        self.freq = freq      # 频率
        self.left = left      # 左子节点
        self.right = right    # 右子节点
    
    def __lt__(self, other):
        return self.freq < other.freq
    
    def is_leaf(self):
        return self.left is None and self.right is None

class LUTHuffmanEncoder:
    """基于LUT分解的Huffman编码器（论文DFloat11方法）"""
    
    def __init__(self, max_code_length=32):
        self.max_code_length = max_code_length
        self.huffman_tree = None
        self.huffman_codes = {}
        
        # 四个分解的LUT，每个256个条目，255作为保留值R
        self.LUT1 = np.full(256, 255, dtype=np.uint8)
        self.LUT2 = np.full(256, 255, dtype=np.uint8) 
        self.LUT3 = np.full(256, 255, dtype=np.uint8)
        self.LUT4 = np.full(256, 255, dtype=np.uint8)
        
        # 编码长度表
        self.code_lengths = np.zeros(256, dtype=np.uint8)
        
    def build_huffman_tree(self, frequencies: Dict[int, float]) -> HuffmanNode:
        """构建Huffman树"""
        heap = []
        
        # 为每个有频率的字符创建叶子节点
        for symbol, freq in frequencies.items():
            if freq > 0:
                node = HuffmanNode(symbol=symbol, freq=freq)
                heapq.heappush(heap, node)
        
        # 特殊情况：只有一个字符
        if len(heap) == 1:
            root = HuffmanNode(freq=heap[0].freq)
            root.left = heapq.heappop(heap)
            return root
        
        # 构建Huffman树
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def generate_huffman_codes(self, root: HuffmanNode) -> Dict[int, str]:
        """生成Huffman编码"""
        codes = {}
        
        def dfs(node: HuffmanNode, code: str):
            if node.is_leaf():
                if code == "":  # 特殊情况：只有一个符号
                    code = "0"
                codes[node.symbol] = code[::-1]
            else:
                if node.left:
                    dfs(node.left, code + "0")
                if node.right:
                    dfs(node.right, code + "1")
        
        dfs(root, "")
        return codes
    
    def adjust_frequencies_for_max_length(self, frequencies: Dict[int, float]) -> Dict[int, float]:
        """调整频率以限制最大编码长度"""
        modified_frequencies = frequencies.copy()
        
        max_iterations = 50  # 防止无限循环
        iteration = 0
        
        while iteration < max_iterations:
            tree = self.build_huffman_tree(modified_frequencies)
            codes = self.generate_huffman_codes(tree)
            
            if not codes:
                break
                
            max_length = max(len(code) for code in codes.values())
            
            if max_length <= self.max_code_length:
                break
            
            # 找到最长编码对应的最不常见的符号，减少其频率
            longest_codes = [(symbol, len(code)) for symbol, code in codes.items() 
                           if len(code) == max_length]
            
            # 按频率排序，选择最不频繁的
            longest_codes.sort(key=lambda x: modified_frequencies[x[0]])
            
            for symbol, _ in longest_codes:
                if modified_frequencies[symbol] > 1:
                    modified_frequencies[symbol] = 1
                    break
            else:
                # 如果所有最长编码的频率都已经是1，强制设置为0.5
                symbol = longest_codes[0][0]
                modified_frequencies[symbol] = 0.5
            
            iteration += 1
        
        assert iteration < max_iterations
        
        return modified_frequencies
    
    def check_and_resolve_lut_conflicts(self, frequencies: Dict[int, float]) -> Dict[int, float]:
        """检查并解决LUT冲突"""
        modified_frequencies = frequencies.copy()
        max_iterations = 20
        iteration = 0
        
        modified_frequencies = self.adjust_frequencies_for_max_length(modified_frequencies)
        tree = self.build_huffman_tree(modified_frequencies)
        codes = self.generate_huffman_codes(tree)
        
        while iteration < max_iterations:
            # 构建临时LUT来检测冲突
            temp_luts = [defaultdict(list) for _ in range(4)]
            
            for symbol, code in codes.items():
                code_len = len(code)
                if code_len == 0:
                    continue
                    
                # # 将编码转换为32位整数
                # code_int = int(code, 2) << (32 - code_len)
                
                # # 提取4个字节
                # byte1 = (code_int >> 24) & 0xFF
                # byte2 = (code_int >> 16) & 0xFF  
                # byte3 = (code_int >> 8) & 0xFF
                # byte4 = code_int & 0xFF
                
                # # 根据编码长度确定应该在哪个LUT中
                # if code_len <= 8:
                #     for i in range(2 ** (8 - code_len)):
                #         temp_luts[0][byte1 + i].append((symbol, code_len))
                # elif code_len <= 16:
                #     for i in range(2 ** (16 - code_len)):
                #         temp_luts[1][byte2 + i].append((symbol, code_len))
                # elif code_len <= 24:
                #     for i in range(2 ** (24 - code_len)):
                #         temp_luts[2][byte3 + i].append((symbol, code_len))
                # else:
                #     for i in range(2 ** (32 - code_len)):
                #         temp_luts[3][byte4 + i].append((symbol, code_len))
                    
                # 将编码转换为32位整数
                code_int = int(code, 2)
                
                # 提取4个字节
                byte4 = (code_int >> 24) & 0xFF
                byte3 = (code_int >> 16) & 0xFF  
                byte2 = (code_int >> 8) & 0xFF
                byte1 = code_int & 0xFF
                
                # 根据编码长度确定应该在哪个LUT中
                if code_len <= 8:
                    for i in range(2 ** (8 - code_len)):
                        temp_luts[0][byte1 + i * 2 ** code_len].append((symbol, code_len))
                elif code_len <= 16:
                    for i in range(2 ** (16 - code_len)):
                        temp_luts[1][byte2 + i * 2 ** (code_len - 8)].append((symbol, code_len))
                elif code_len <= 24:
                    for i in range(2 ** (24 - code_len)):
                        temp_luts[2][byte3 + i * 2 ** (code_len - 16)].append((symbol, code_len))
                else:
                    for i in range(2 ** (32 - code_len)):
                        temp_luts[3][byte4 + i * 2 ** (code_len - 24)].append((symbol, code_len))
            
            # 检查冲突
            conflicts = []
            for lut_idx, lut in enumerate(temp_luts):
                for byte_val, symbols in lut.items():
                    if len(symbols) > 1:
                        conflicts.append((lut_idx, byte_val, symbols))
            
            if not conflicts:
                break  # 没有冲突
            
            # 解决冲突：增加更频繁符号的频率来缩短其编码
            for lut_idx, byte_val, symbols in conflicts:
                # 按频率排序，选择最频繁的符号
                symbols.sort(key=lambda x: modified_frequencies[x[0]], reverse=True)
                most_frequent_symbol = symbols[0][0]
                
                # 增加最频繁符号的频率
                current_freq = modified_frequencies[most_frequent_symbol]
                modified_frequencies[most_frequent_symbol] = current_freq * 1.5
            
            # 重新构建树和编码
            modified_frequencies = self.adjust_frequencies_for_max_length(modified_frequencies)
            tree = self.build_huffman_tree(modified_frequencies)
            codes = self.generate_huffman_codes(tree)
            iteration += 1
        
        assert iteration < max_iterations
        
        return modified_frequencies
    
    def build_decomposed_luts(self, codes: Dict[int, str]):
        """构建分解的LUT表"""
        # 清空LUT
        self.LUT1.fill(255)
        self.LUT2.fill(255) 
        self.LUT3.fill(255)
        self.LUT4.fill(255)
        self.code_lengths.fill(0)
        
        for symbol, code in codes.items():
            code_len = len(code)
            self.code_lengths[symbol] = code_len
            
            if code_len == 0:
                continue
            
            # # 将二进制字符串转换为32位整数
            # code_int = int(code, 2) << (32 - code_len)
            
            # # 提取4个字节
            # byte1 = (code_int >> 24) & 0xFF
            # byte2 = (code_int >> 16) & 0xFF
            # byte3 = (code_int >> 8) & 0xFF
            # byte4 = code_int & 0xFF
            
            # # 根据编码长度在相应的LUT中设置值
            # if code_len <= 8:
            #     for i in range(2 ** (8 - code_len)):
            #         self.LUT1[byte1 + i] = symbol
            # elif code_len <= 16:
            #     for i in range(2 ** (16 - code_len)):
            #         self.LUT2[byte2 + i] = symbol
            # elif code_len <= 24:
            #     for i in range(2 ** (24 - code_len)):
            #         self.LUT3[byte3 + i] = symbol
            # else:
            #     for i in range(2 ** (32 - code_len)):
            #         self.LUT4[byte4 + i] = symbol
                    
            # 将编码转换为32位整数
            code_int = int(code, 2)
            
            # 提取4个字节
            byte4 = (code_int >> 24) & 0xFF
            byte3 = (code_int >> 16) & 0xFF  
            byte2 = (code_int >> 8) & 0xFF
            byte1 = code_int & 0xFF
            
            # 根据编码长度确定应该在哪个LUT中
            if code_len <= 8:
                for i in range(2 ** (8 - code_len)):
                    self.LUT1[byte1 + i * 2 ** code_len] = symbol
            elif code_len <= 16:
                for i in range(2 ** (16 - code_len)):
                    self.LUT2[byte2 + i * 2 ** (code_len - 8)] = symbol
            elif code_len <= 24:
                for i in range(2 ** (24 - code_len)):
                    self.LUT3[byte3 + i * 2 ** (code_len - 16)] = symbol
            else:
                for i in range(2 ** (32 - code_len)):
                    self.LUT4[byte4 + i * 2 ** (code_len - 24)] = symbol
    
    @staticmethod
    @jit(nopython=True)
    def encode_to_bitstream_jit(data: np.ndarray, codes: np.ndarray) -> str:
        """编码数据为位流"""
        bit_string = ""
        for byte_val in data:
            bit_string += codes[byte_val]
        
        return bit_string
    
    @staticmethod
    def encode_to_bitstream(data: np.ndarray, codes: Dict[int, str], progress=True) -> str:
        import huffman_encode
        return huffman_encode.sum_as_string(data, codes)
        """编码数据为位流"""
        bit_string = ""
        for byte_val in tqdm(data, disable=not progress):
            if byte_val in codes:
                bit_string += codes[byte_val][::-1]
                # bit_string = codes[byte_val] + bit_string
            else:
                raise ValueError(f"字节值 {byte_val} 不在编码表中")
        
        return bit_string[::-1]
    
    def decode_with_lut(self, bit_string: str, num_elements: int) -> List[int]:
        """使用LUT解码位流"""
        decoded_data = []
        bit_offset = 0
        
        bit_string = '1' * 32 + bit_string
        
        bar = tqdm(total=num_elements)
        while len(decoded_data) < num_elements and bit_offset < len(bit_string):
            # 读取32位（如果可用）
            remaining_bits = len(bit_string) - bit_offset
            if remaining_bits < 8:
                break
                
            # 提取最多32位
            read_bits = min(32, remaining_bits)
            bits_segment = bit_string[-(bit_offset + read_bits):len(bit_string)-bit_offset]
            
            # 填充到32位
            if len(bits_segment) < 32:
                bits_segment = bits_segment.rjust(32, '0')
            
            # 转换为4个字节
            code_int = int(bits_segment, 2)
            byte4 = (code_int >> 24) & 0xFF
            byte3 = (code_int >> 16) & 0xFF
            byte2 = (code_int >> 8) & 0xFF
            byte1 = code_int & 0xFF
            
            # 使用LUT解码
            decoded_symbol = None
            code_length = 0
            
            # 依次尝试4个LUT
            if self.LUT1[byte1] != 255:
                decoded_symbol = self.LUT1[byte1]
                code_length = self.code_lengths[decoded_symbol]
            elif self.LUT2[byte2] != 255:
                decoded_symbol = self.LUT2[byte2]
                code_length = self.code_lengths[decoded_symbol]
            elif self.LUT3[byte3] != 255:
                decoded_symbol = self.LUT3[byte3]
                code_length = self.code_lengths[decoded_symbol]
            elif self.LUT4[byte4] != 255:
                decoded_symbol = self.LUT4[byte4]
                code_length = self.code_lengths[decoded_symbol]
            
            if decoded_symbol is not None:
                decoded_data.append(decoded_symbol)
                bit_offset += code_length
            else:
                # 解码失败，跳过一位
                bit_offset += 1
                assert 0
            
            bar.update(1)
        
        return decoded_data
    
    def build_lut(self, frequencies: Union[np.ndarray, torch.Tensor]):
        """编码数据"""
        # 转换输入
        if torch.is_tensor(frequencies):
            frequencies = frequencies.cpu().numpy()
        
        # 构建频率字典
        freq_dict = {}
        for i in range(256):
            if frequencies[i] > 0:
                freq_dict[i] = float(frequencies[i])
        
        # 解决LUT冲突
        final_frequencies = self.check_and_resolve_lut_conflicts(freq_dict)
        
        # 重新构建最终的树和编码
        self.huffman_tree = self.build_huffman_tree(final_frequencies)
        self.huffman_codes = self.generate_huffman_codes(self.huffman_tree)
        
        # 构建分解的LUT
        self.build_decomposed_luts(self.huffman_codes)
        
    
    def encode(self, data: Union[np.ndarray, torch.Tensor], 
               frequencies: Union[np.ndarray, torch.Tensor]) -> Tuple[str, Dict]:
        """编码数据"""
        # 转换输入
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        data = data.flatten().astype(np.uint8)
        
        self.build_lut(frequencies)
        
        jit_codes = np.array([self.huffman_codes.get(i, '') for i in range(256)], dtype=str)
        print(jit_codes)
        
        # 编码数据
        encoded_bitstring = self.encode_to_bitstream(data, self.huffman_codes)
        # encoded_bitstring = self.encode_to_bitstream_jit(data, jit_codes)
        
        # 计算统计信息
        original_bits = len(data) * 8
        compressed_bits = len(encoded_bitstring)
        compression_ratio = compressed_bits / original_bits if original_bits > 0 else 0
        
        stats = {
            "original_bytes": len(data),
            "original_bits": original_bits,
            "compressed_bits": compressed_bits,
            "compression_ratio": compression_ratio,
            "space_saving_percentage": (1 - compression_ratio) * 100,
            "huffman_codes": self.huffman_codes.copy(),
            "average_code_length": compressed_bits / len(data) if len(data) > 0 else 0,
            "max_code_length": max(len(code) for code in self.huffman_codes.values()) if self.huffman_codes else 0,
            "lut_memory_usage": 4 * 256 + 256  # 4个LUT + code_lengths表
        }
        
        return encoded_bitstring, stats
    
    def decode(self, encoded_bitstring: str, original_length: int) -> List[int]:
        """解码位流"""
        return self.decode_with_lut(encoded_bitstring, original_length)

def lut_huffman_compress(data: Union[np.ndarray, torch.Tensor], 
                        frequencies: Union[np.ndarray, torch.Tensor]) -> Tuple[str, Dict]:
    """
    基于LUT分解的Huffman压缩函数（DFloat11风格）
    
    Args:
        data: 字节数组（Tensor或numpy数组）
        frequencies: 256个字节值的频率（Tensor或numpy数组）
    
    Returns:
        压缩后的二进制字符串和统计信息
    """
    encoder = LUTHuffmanEncoder()
    return encoder.encode(data, frequencies)

def demonstrate_lut_huffman():
    """演示LUT Huffman编码"""
    print("基于LUT分解的Huffman编码演示")
    print("=" * 50)
    
    # 创建模拟的字节数据，具有不均匀分布（类似BFloat16指数）
    np.random.seed(42)
    
    # 生成具有高度不均匀分布的数据
    data = []
    # 大多数值集中在几个常见字节
    common_values = [120, 121, 122, 123, 119, 124, 118, 125]
    for _ in range(8000):
        if np.random.random() < 0.8:
            data.append(np.random.choice(common_values))
        else:
            data.append(np.random.randint(0, 256))
    
    data = np.array(data, dtype=np.uint8)
    
    # 计算频率
    frequencies = np.zeros(256, dtype=np.float32)
    unique, counts = np.unique(data, return_counts=True)
    frequencies[unique] = counts / len(data)
    
    print(f"数据长度: {len(data)} 字节")
    print(f"使用的不同字节值: {len(unique)}")
    
    # 使用LUT Huffman编码
    encoder = LUTHuffmanEncoder()
    encoded_string, stats = encoder.encode(data, frequencies)
    
    print(f"\n编码统计信息:")
    print("-" * 30)
    print(f"原始位数: {stats['original_bits']}")
    print(f"压缩后位数: {stats['compressed_bits']}")
    print(f"压缩率: {stats['compression_ratio']:.4f}")
    print(f"节省空间: {stats['space_saving_percentage']:.2f}%")
    print(f"平均编码长度: {stats['average_code_length']:.2f} 位/字节")
    print(f"最大编码长度: {stats['max_code_length']} 位")
    print(f"LUT内存使用: {stats['lut_memory_usage']} 字节")
    
    # 显示一些编码
    print(f"\n常用字节的Huffman编码:")
    print("-" * 30)
    for byte_val in common_values:
        if byte_val in encoder.huffman_codes:
            print(f"字节 {byte_val:3d}: {encoder.huffman_codes[byte_val]} ({len(encoder.huffman_codes[byte_val])} 位)")
    
    # 验证解码
    decoded_data = encoder.decode(encoded_string, len(data))
    
    print(f"\n解码验证:")
    print("-" * 20)
    is_correct = (decoded_data == data.tolist())
    print(f"解码长度: {len(decoded_data)}")
    print(f"解码正确: {'是' if is_correct else '否'}")
    
    if not is_correct:
        diff_count = sum(1 for i, (a, b) in enumerate(zip(decoded_data, data)) if a != b)
        print(f"错误数量: {diff_count}")
    
    # LUT使用统计
    lut_usage = [
        np.sum(encoder.LUT1 != 255),
        np.sum(encoder.LUT2 != 255), 
        np.sum(encoder.LUT3 != 255),
        np.sum(encoder.LUT4 != 255)
    ]
    
    print(f"\nLUT使用统计:")
    print("-" * 20)
    for i, usage in enumerate(lut_usage, 1):
        print(f"LUT{i}: {usage}/256 条目被使用")
    
    return encoder, stats

if __name__ == "__main__":
    # 运行演示
    demonstrate_lut_huffman()
    
    print(f"\n{'='*50}")
    print("LUT分解Huffman编码实现完成!")
    print("关键特性:")
    print("• 最大编码长度限制 (32位)")
    print("• LUT冲突检测和解决")
    print("• 频率调整算法") 
    print("• 分解LUT设计 (4×256条目)")
    print("• 高效解码算法")