"""
工具检索器（Tool Retrieval / RAG-over-Tools）。

为什么用字符 n-gram + TF-IDF：
    - 工具描述以中文为主，中文无天然空格分词。字符 bigram/trigram 既不需要
      jieba 等外部分词器，又能捕捉到"网页搜索""数学计算"这类高频组合。
    - 纯标准库实现，便于教学时逐行讲解 TF-IDF 的工作原理。
    - 生产场景可以直接把 TfidfRetriever 替换为基于 sentence-transformers
      或 OpenAI embeddings 的实现，ToolRegistry 的接口保持不变。

公共接口（其它 retriever 实现也应遵循）：
    - index(specs: list[ToolSpec]) -> None     # 重建索引
    - search(query: str, k: int) -> list[ToolSpec]   # 返回 Top-K
"""

import math
from collections import Counter
from typing import List, Dict, Tuple


def _tokenize(text: str, n_min: int = 2, n_max: int = 3) -> List[str]:
    """
    混合分词：
    - ASCII 字母数字按"单词"切（连续 [a-z0-9]）
    - 非 ASCII（主要是中文）按字符 n-gram 切（n=2,3）
    """
    text = text.lower()
    tokens: List[str] = []

    # 1) ASCII 单词
    buf: List[str] = []
    for ch in text:
        if ch.isalnum() and ord(ch) < 128:
            buf.append(ch)
        else:
            if buf:
                tokens.append("".join(buf))
                buf.clear()
    if buf:
        tokens.append("".join(buf))

    # 2) 非 ASCII 字符 n-gram
    non_ascii = "".join(ch for ch in text if ord(ch) >= 128)
    for n in range(n_min, n_max + 1):
        for i in range(len(non_ascii) - n + 1):
            tokens.append(non_ascii[i : i + n])

    return tokens


class TfidfRetriever:
    """字符 n-gram TF-IDF 检索器。"""

    def __init__(self):
        self.idf: Dict[str, float] = {}
        self.doc_vecs: List[Tuple[str, Dict[str, float]]] = []  # (name, L2 归一化向量)
        self.specs_by_name: Dict[str, object] = {}

    def index(self, specs) -> None:
        """对工具规范列表重建 TF-IDF 索引。"""
        self.specs_by_name = {s.name: s for s in specs}
        docs = [_tokenize(s.index_text()) for s in specs]
        n_docs = len(docs) or 1

        df: Counter = Counter()
        for tokens in docs:
            df.update(set(tokens))

        # 平滑 IDF，避免高频词权重为 0、零样本时除零
        self.idf = {
            t: math.log((n_docs + 1) / (c + 1)) + 1.0 for t, c in df.items()
        }

        self.doc_vecs = []
        for spec, tokens in zip(specs, docs):
            self.doc_vecs.append((spec.name, self._vectorize(tokens)))

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        """TF * IDF 并做 L2 归一化，得到稀疏向量。"""
        tf = Counter(tokens)
        vec = {t: c * self.idf.get(t, 0.0) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {t: v / norm for t, v in vec.items()}

    def search(self, query: str, k: int = 8) -> List:
        """返回与 query 余弦相似度最高的前 k 个 ToolSpec。"""
        if not self.doc_vecs:
            return []

        q_vec = self._vectorize(_tokenize(query))
        scored: List[Tuple[float, str]] = []
        for name, d_vec in self.doc_vecs:
            # 走 query 的稀疏侧累加，复杂度 O(|q|)
            score = sum(w * d_vec.get(t, 0.0) for t, w in q_vec.items())
            scored.append((score, name))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [self.specs_by_name[name] for _, name in scored[:k]]


if __name__ == "__main__":
    # 简单自测：构造若干 mock spec，确认排序符合直觉
    from dataclasses import dataclass

    @dataclass
    class MockSpec:
        name: str
        text: str
        def index_text(self): return self.text

    specs = [
        MockSpec("Search", "网页搜索引擎 时事 新闻 事实"),
        MockSpec("Calculator", "数学计算 加减乘除 sqrt sin cos"),
        MockSpec("FileRead", "读取本地文件 文本内容"),
        MockSpec("Translate", "中英互译 翻译"),
    ]
    r = TfidfRetriever()
    r.index(specs)

    for q in ["最新的GPU新闻", "计算 sqrt(2)*pi", "把这段话翻译成英文"]:
        top = r.search(q, k=2)
        print(f"Query: {q}  ->  {[s.name for s in top]}")
