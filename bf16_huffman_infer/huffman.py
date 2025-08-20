import numpy as np
import heapq
from typing import Dict, List, Tuple, Union
import torch
from collections import defaultdict

from tqdm import tqdm

class HuffmanNode:
    """Huffman tree node"""
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol  # symbol (byte value 0-255)
        self.freq = freq      # frequency
        self.left = left      # left child node
        self.right = right    # right child node
    
    def __lt__(self, other):
        return self.freq < other.freq
    
    def is_leaf(self):
        return self.left is None and self.right is None

class LUTHuffmanEncoder:
    """LUT decomposition-based Huffman encoder (DFloat11 method from paper)"""
    
    def __init__(self, max_code_length=32):
        self.max_code_length = max_code_length
        self.huffman_tree = None
        self.huffman_codes = {}
        
        # Four decomposed LUTs, each with 256 entries, 255 as reserved value R
        self.LUT1 = np.full(256, 255, dtype=np.uint8)
        self.LUT2 = np.full(256, 255, dtype=np.uint8) 
        self.LUT3 = np.full(256, 255, dtype=np.uint8)
        self.LUT4 = np.full(256, 255, dtype=np.uint8)
        
        # Code length table
        self.code_lengths = np.zeros(256, dtype=np.uint8)
        
    def build_huffman_tree(self, frequencies: Dict[int, float]) -> HuffmanNode:
        """Build Huffman tree"""
        heap = []
        
        # Create leaf nodes for each character with frequency
        for symbol, freq in frequencies.items():
            if freq > 0:
                node = HuffmanNode(symbol=symbol, freq=freq)
                heapq.heappush(heap, node)
        
        # Special case: only one character
        if len(heap) == 1:
            root = HuffmanNode(freq=heap[0].freq)
            root.left = heapq.heappop(heap)
            return root
        
        # Build Huffman tree
        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            
            merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
            heapq.heappush(heap, merged)
        
        return heap[0]
    
    def generate_huffman_codes(self, root: HuffmanNode) -> Dict[int, str]:
        """Generate Huffman codes"""
        codes = {}
        
        def dfs(node: HuffmanNode, code: str):
            if node.is_leaf():
                if code == "":  # Special case: only one symbol
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
        """Adjust frequencies to limit maximum code length"""
        modified_frequencies = frequencies.copy()
        
        max_iterations = 50  # Prevent infinite loop
        iteration = 0
        
        while iteration < max_iterations:
            tree = self.build_huffman_tree(modified_frequencies)
            codes = self.generate_huffman_codes(tree)
            
            if not codes:
                break
                
            max_length = max(len(code) for code in codes.values())
            
            if max_length <= self.max_code_length:
                break
            
            # Find the least common symbol corresponding to the longest code, reduce its frequency
            longest_codes = [(symbol, len(code)) for symbol, code in codes.items() 
                           if len(code) == max_length]
            
            # Sort by frequency, select the least frequent
            longest_codes.sort(key=lambda x: modified_frequencies[x[0]])
            
            for symbol, _ in longest_codes:
                if modified_frequencies[symbol] > 1:
                    modified_frequencies[symbol] = 1
                    break
            else:
                # If all longest codes have frequency 1, force set to 0.5
                symbol = longest_codes[0][0]
                modified_frequencies[symbol] = 0.5
            
            iteration += 1
        
        assert iteration < max_iterations
        
        return modified_frequencies
    
    def check_and_resolve_lut_conflicts(self, frequencies: Dict[int, float]) -> Dict[int, float]:
        """Check and resolve LUT conflicts"""
        modified_frequencies = frequencies.copy()
        max_iterations = 20
        iteration = 0
        
        modified_frequencies = self.adjust_frequencies_for_max_length(modified_frequencies)
        tree = self.build_huffman_tree(modified_frequencies)
        codes = self.generate_huffman_codes(tree)
        
        while iteration < max_iterations:
            # Build temporary LUTs to detect conflicts
            temp_luts = [defaultdict(list) for _ in range(4)]
            
            for symbol, code in codes.items():
                code_len = len(code)
                if code_len == 0:
                    continue
                    
                # # Convert encoding to 32-bit integer
                # code_int = int(code, 2) << (32 - code_len)
                
                # # Extract 4 bytes
                # byte1 = (code_int >> 24) & 0xFF
                # byte2 = (code_int >> 16) & 0xFF  
                # byte3 = (code_int >> 8) & 0xFF
                # byte4 = code_int & 0xFF
                
                # # Determine which LUT to use based on encoding length
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
                    
                # Convert encoding to 32-bit integer
                code_int = int(code, 2)
                
                # Extract 4 bytes
                byte4 = (code_int >> 24) & 0xFF
                byte3 = (code_int >> 16) & 0xFF  
                byte2 = (code_int >> 8) & 0xFF
                byte1 = code_int & 0xFF
                
                # Determine which LUT to use based on encoding length
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
            
            # Check for conflicts
            conflicts = []
            for lut_idx, lut in enumerate(temp_luts):
                for byte_val, symbols in lut.items():
                    if len(symbols) > 1:
                        conflicts.append((lut_idx, byte_val, symbols))
            
            if not conflicts:
                break  # No conflicts
            
            # Resolve conflicts: increase frequency of more frequent symbols to shorten their encoding
            for lut_idx, byte_val, symbols in conflicts:
                # Sort by frequency, select the most frequent symbol
                symbols.sort(key=lambda x: modified_frequencies[x[0]], reverse=True)
                most_frequent_symbol = symbols[0][0]
                
                # Increase frequency of the most frequent symbol
                current_freq = modified_frequencies[most_frequent_symbol]
                modified_frequencies[most_frequent_symbol] = current_freq * 1.5
            
            # Rebuild tree and codes
            modified_frequencies = self.adjust_frequencies_for_max_length(modified_frequencies)
            tree = self.build_huffman_tree(modified_frequencies)
            codes = self.generate_huffman_codes(tree)
            iteration += 1
        
        assert iteration < max_iterations
        
        return modified_frequencies
    
    def build_decomposed_luts(self, codes: Dict[int, str]):
        """Build decomposed LUT tables"""
        # Clear LUTs
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
            
            # # Convert binary string to 32-bit integer
            # code_int = int(code, 2) << (32 - code_len)
            
            # # Extract 4 bytes
            # byte1 = (code_int >> 24) & 0xFF
            # byte2 = (code_int >> 16) & 0xFF
            # byte3 = (code_int >> 8) & 0xFF
            # byte4 = code_int & 0xFF
            
            # # Set values in corresponding LUT based on encoding length
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
                    
            # Convert encoding to 32-bit integer
            code_int = int(code, 2)
            
            # Extract 4 bytes
            byte4 = (code_int >> 24) & 0xFF
            byte3 = (code_int >> 16) & 0xFF  
            byte2 = (code_int >> 8) & 0xFF
            byte1 = code_int & 0xFF
            
            # Determine which LUT to use based on encoding length
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
    def encode_to_bitstream(data: np.ndarray, codes: Dict[int, str], progress=True) -> str:
        """Encode data to bitstream"""
        bit_string = ""
        for byte_val in tqdm(data, disable=not progress):
            if byte_val in codes:
                bit_string += codes[byte_val][::-1]
                # bit_string = codes[byte_val] + bit_string
            else:
                raise ValueError(f"Byte value {byte_val} not in encoding table")
        
        return bit_string[::-1]
    
    def decode_with_lut(self, bit_string: str, num_elements: int) -> List[int]:
        """Decode bitstream using LUT"""
        decoded_data = []
        bit_offset = 0
        
        bit_string = '1' * 32 + bit_string
        
        bar = tqdm(total=num_elements)
        while len(decoded_data) < num_elements and bit_offset < len(bit_string):
            remaining_bits = len(bit_string) - bit_offset
            if remaining_bits < 8:
                break
                
            read_bits = min(32, remaining_bits)
            bits_segment = bit_string[-(bit_offset + read_bits):len(bit_string)-bit_offset]
            
            if len(bits_segment) < 32:
                bits_segment = bits_segment.rjust(32, '0')
            
            code_int = int(bits_segment, 2)
            byte4 = (code_int >> 24) & 0xFF
            byte3 = (code_int >> 16) & 0xFF
            byte2 = (code_int >> 8) & 0xFF
            byte1 = code_int & 0xFF
            
            decoded_symbol = None
            code_length = 0
            
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
                bit_offset += 1
                assert 0
            
            bar.update(1)
        
        return decoded_data
    
    def build_lut(self, frequencies: Union[np.ndarray, torch.Tensor]):
        if torch.is_tensor(frequencies):
            frequencies = frequencies.cpu().numpy()
        
        freq_dict = {}
        for i in range(256):
            if frequencies[i] > 0:
                freq_dict[i] = float(frequencies[i])
        
        final_frequencies = self.check_and_resolve_lut_conflicts(freq_dict)
        
        self.huffman_tree = self.build_huffman_tree(final_frequencies)
        self.huffman_codes = self.generate_huffman_codes(self.huffman_tree)
        
        self.build_decomposed_luts(self.huffman_codes)
        
    
    def encode(self, data: Union[np.ndarray, torch.Tensor], 
               frequencies: Union[np.ndarray, torch.Tensor]) -> Tuple[str, Dict]:
        if torch.is_tensor(data):
            data = data.cpu().numpy()
        data = data.flatten().astype(np.uint8)
        
        self.build_lut(frequencies)
        
        jit_codes = np.array([self.huffman_codes.get(i, '') for i in range(256)], dtype=str)
        print(jit_codes)
        
        encoded_bitstring = self.encode_to_bitstream(data, self.huffman_codes)
        # encoded_bitstring = self.encode_to_bitstream_jit(data, jit_codes)
        
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
            "lut_memory_usage": 4 * 256 + 256  # 4 LUTs + code_lengths table
        }
        
        return encoded_bitstring, stats
    
    def decode(self, encoded_bitstring: str, original_length: int) -> List[int]:
        return self.decode_with_lut(encoded_bitstring, original_length)
