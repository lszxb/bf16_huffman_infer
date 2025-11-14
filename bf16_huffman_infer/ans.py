from collections import Counter
from typing import Dict, List, Mapping, Sequence


class RANSEncoder:
    def __init__(self, freq: Mapping[int, int], precision: int = 12):
        """
        Args:
            freq: 符号 -> 频次，符号需为 0..255 的整数。
            precision: LUT 精度位数，推荐 [8, 15]。
        """
        if precision < 1 or precision > 15:
            raise ValueError("precision 必须位于 [1, 15]")
        if not freq:
            raise ValueError("freq 不能为空")
        
        if isinstance(freq, list):
            freq = {s: f for s, f in enumerate(freq)}

        self.precision = precision
        self.base = 1 << precision
        self.freq = self._normalize(freq)
        self.freq = [self.freq.get(s, 0) for s in range(16)]
        self.mask = self.base - 1
        
        self.cum = []
        acc = 0
        for f in self.freq:
            self.cum.append(acc)
            acc += f

        self.decode_lut = []
        for s, f in enumerate(self.freq):
            self.decode_lut += zip([s] * f, [f] * f, range(f))

    @classmethod
    def from_data(cls, data: Sequence[int], precision: int = 12) -> "RANSEncoder":
        """根据样本估计频率并创建编码器。"""
        counts = Counter(int(x) for x in data)
        return cls(counts, precision)

    def _normalize(self, freq: Mapping[int, int]) -> Dict[int, int]:
        total = sum(freq.values())
        if total <= 0:
            raise ValueError("频率之和必须大于 0")

        scaled: Dict[int, int] = {}
        remainders = []
        for sym, count in freq.items():
            if count <= 0:
                raise ValueError(f"符号 {sym} 的频率必须 > 0")
            value = count * self.base / total
            base_val = int(value)
            if base_val == 0:
                base_val = 1
            scaled[sym] = base_val
            remainders.append((value - base_val, sym))

        current = sum(scaled.values())
        if current < self.base:
            remainders.sort(reverse=True)
            for _, sym in remainders:
                if current == self.base:
                    break
                scaled[sym] += 1
                current += 1
        elif current > self.base:
            remainders.sort()
            for _, sym in remainders:
                if current == self.base:
                    break
                if scaled[sym] > 1:
                    scaled[sym] -= 1
                    current -= 1

        if sum(scaled.values()) != self.base:
            raise ValueError("归一化失败，无法匹配目标基数")
        return scaled

    def compress(self, symbols: Sequence[int]) -> bytes:
        """
        压缩符号序列。
        Args:
            symbols: 可迭代的 0..255 整数序列。
        Returns:
            bytes: 压缩后的二进制流。
        """
        seq = [int(x) for x in symbols]
        state = self.base
        out = bytearray()

        for sym in reversed(seq):
            freq = self.freq[sym]
            start = self.cum[sym]

            while state >= (freq << 32):
                out += (state & 0xFFFFFFFF).to_bytes(length=4)
                state >>= 32

            quotient, remainder = divmod(state, freq)
            state = (quotient << self.precision) + remainder + start
            # print(sym, state, out)

        out += (state & 0xFFFFFFFF).to_bytes(length=4)
        state >>= 32
        out += (state & 0xFFFFFFFF).to_bytes(length=4)
        state >>= 32
        
        # out.extend(state.to_bytes(self._STATE_BYTES))
        return bytes(out)

    def decompress(self, blob: bytes, output_len: int) -> List[int]:
        """
        解压缩符号序列。
        Args:
            blob: compress 的返回值。
            output_len: 期望输出的符号数量。
        Returns:
            List[int]: 解码后的符号。
        """
        if output_len == 0:
            return []

        buffer = blob

        ptr = len(buffer)
        state = 0

        result: List[int] = []
        for _ in range(output_len):
            while state < self.base:
                if ptr == 0:
                    raise ValueError("压缩流损坏或长度不足")
                ptr -= 4
                state = (state << 32) | int.from_bytes(buffer[ptr:ptr+4])
            
            slot = state & self.mask
            sym, freq, rem = self.decode_lut[slot]
            result.append(sym)

            state = freq * (state >> self.precision) + rem
            
            # print(sym, state, buffer[:ptr])

        return result


# --- 使用示例 ---
if __name__ == "__main__":
    sample = [1, 2, 3, 3, 2, 12, 13, 11, 1, 8, 1, 7, 1, 6, 1, 1]
    codec = RANSEncoder.from_data(sample, precision=12)

    packed = codec.compress(sample)
    restored = codec.decompress(packed, len(sample))

    print(f"原始长度: {len(sample)}")
    print(f"压缩后长度: {len(packed)} 字节")
    print("是否还原一致:", restored == sample)
    print(restored)