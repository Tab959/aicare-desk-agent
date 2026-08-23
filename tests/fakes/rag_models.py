"""提供可控制维度、数值、延迟和相关性的BGE测试替身。"""

from __future__ import annotations

import math
import time
from typing import Any


class FakeTokenizer:
    """按字符生成token，并支持真实适配器所用的截断与解码接口。"""

    def __call__(self, text: str, **kwargs: Any) -> dict[str, list[Any]]:
        maximum = int(kwargs.get("max_length", len(text)))
        truncated = text[:maximum] if kwargs.get("truncation") else text
        return {
            "input_ids": [ord(char) for char in truncated],
            "offset_mapping": [(index, index + 1) for index in range(len(truncated))],
        }

    def decode(self, token_ids: list[int], **_: Any) -> str:
        return "".join(chr(token_id) for token_id in token_ids)


class FakeEmbeddingModel:
    """生成固定维度向量，并可注入畸形维度、非有限值或同步延迟。"""

    def __init__(
        self, *, dimensions: int = 1024, invalid_number: float | None = None, delay: float = 0
    ) -> None:
        self.dimensions = dimensions
        self.invalid_number = invalid_number
        self.delay = delay
        self.tokenizer = FakeTokenizer()

    def encode(self, texts: list[str], **_: Any) -> dict[str, list[list[float]]]:
        if self.delay:
            time.sleep(self.delay)
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dimensions
            vector[0] = float(max(len(text), 1))
            if self.dimensions > 1:
                vector[1] = 1.0
            if self.invalid_number is not None:
                vector[-1] = self.invalid_number
            vectors.append(vector)
        return {"dense_vecs": vectors}


class FakeRerankerModel:
    """按query与passage的字符交集打分，并记录是否出现超长passage。"""

    def __init__(self, *, invalid_number: float | None = None, delay: float = 0) -> None:
        self.invalid_number = invalid_number
        self.delay = delay
        self.tokenizer = FakeTokenizer()

    def compute_score(self, pairs: list[list[str]], **_: Any) -> list[float]:
        if self.delay:
            time.sleep(self.delay)
        if self.invalid_number is not None:
            return [self.invalid_number for _ in pairs]
        return [float(len(set(query) & set(passage))) for query, passage in pairs]


def non_finite_values() -> tuple[float, ...]:
    """返回模型输出禁止出现的全部IEEE特殊值。"""
    return (math.nan, math.inf, -math.inf)
