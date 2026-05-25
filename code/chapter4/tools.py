from dotenv import load_dotenv
load_dotenv()

import os
import math
from dataclasses import dataclass, field
from typing import Callable, Any, List, Dict, Optional

from serpapi import SerpApiClient


# ====================================================================
# 1. 工具实现（函数本体）
# ====================================================================

def search(query: str) -> str:
    """
    一个基于 SerpApi 的网页搜索引擎工具。
    它会智能地解析搜索结果，优先返回直接答案或知识图谱信息。
    """
    print(f"🔍 正在执行 [SerpApi] 网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "错误：SERPAPI_API_KEY 未在 .env 文件中配置。"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",
            "hl": "zh-cn",
        }
        client = SerpApiClient(params)
        results = client.get_dict()

        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)

        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"


def calculator(expression: str) -> str:
    """
    一个安全的数学计算器工具。
    支持加减乘除、括号、幂运算、以及常用数学函数（sqrt、sin、cos 等）。
    """
    print(f"🧮 正在执行计算: {expression}")
    try:
        safe_globals = {"__builtins__": {}}
        safe_locals = {
            "sqrt": math.sqrt, "pow": math.pow, "abs": abs, "round": round,
            "log": math.log, "log10": math.log10, "log2": math.log2, "exp": math.exp,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "asin": math.asin, "acos": math.acos, "atan": math.atan,
            "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil,
        }

        forbidden_keywords = ["import", "exec", "eval", "open", "os", "sys", "__"]
        for keyword in forbidden_keywords:
            if keyword in expression.lower():
                return f"错误：表达式包含不允许的关键字 '{keyword}'。"

        result = eval(expression, safe_globals, safe_locals)  # noqa: S307

        if isinstance(result, float) and result.is_integer():
            return f"计算结果：{expression} = {int(result)}"
        return f"计算结果：{expression} = {result}"

    except ZeroDivisionError:
        return "错误：除数不能为零。"
    except SyntaxError:
        return f"错误：表达式语法有误，请检查 '{expression}'。"
    except Exception as e:
        return f"计算时发生错误: {e}"


# ====================================================================
# 2. 结构化工具规范（Layer 1）
# ====================================================================

@dataclass
class ToolSpec:
    """
    一个工具的完整规范。相比"自由文本描述"，结构化字段让 LLM
    更容易做出正确的选择，也让代码侧能在调用前完成参数校验。
    """
    name: str
    description: str                       # 一句话用途
    category: str                          # 领域分类，如 search / math / code / io
    parameters: Dict[str, Any]             # JSON Schema（OpenAI function calling 风格）
    when_to_use: str                       # 适用场景
    when_not_to_use: str                   # 反例（关键：减少误用）
    func: Callable                         # 实际执行函数
    examples: List[Dict[str, str]] = field(default_factory=list)

    def index_text(self) -> str:
        """供检索使用的拼接文本（包含会影响相关性的所有字段）。"""
        ex = " ".join(e.get("input", "") for e in self.examples)
        return f"{self.name} {self.description} {self.when_to_use} {self.when_not_to_use} {ex}"

    def format(self) -> str:
        """供 prompt 注入的格式化描述。"""
        props = self.parameters.get("properties", {})
        param_list = ", ".join(props.keys()) if props else "无"
        lines = [
            f"- **{self.name}** [{self.category}]: {self.description}",
            f"    参数: {param_list}",
            f"    适用: {self.when_to_use}",
            f"    不适用: {self.when_not_to_use}",
        ]
        if self.examples:
            ex = self.examples[0]
            lines.append(f"    示例: {self.name}[{ex.get('input', '')}]")
        return "\n".join(lines)


# ====================================================================
# 3. 工具注册表（Layer 1 + Layer 2 入口）
# ====================================================================

class ToolRegistry:
    """
    支持结构化规范注册 + Top-K 检索的工具注册表。

    与旧版 ToolExecutor 的差异：
    - 接收 ToolSpec 而不是 (name, description, func) 三元组。
    - 提供 retrieve(query, k)：当工具数量较多时，按相关性召回子集。
    - 提供 format_for_prompt()：把召回结果格式化为 prompt 片段。
    """

    def __init__(self, retriever: Optional[Any] = None):
        self.specs: Dict[str, ToolSpec] = {}
        self.retriever = retriever  # 可选；通常是 TfidfRetriever 实例

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            print(f"警告：工具 '{spec.name}' 已存在，将被覆盖。")
        self.specs[spec.name] = spec
        # 每次注册都重建索引，工具量不大时开销可以忽略
        if self.retriever is not None:
            self.retriever.index(list(self.specs.values()))
        print(f"工具 '{spec.name}' 已注册（分类: {spec.category}）。")

    def get(self, name: str) -> Optional[Callable]:
        spec = self.specs.get(name)
        return spec.func if spec else None

    def all(self) -> List[ToolSpec]:
        return list(self.specs.values())

    def names(self) -> List[str]:
        return list(self.specs.keys())

    def retrieve(self, query: str, k: int = 8) -> List[ToolSpec]:
        """
        按 query 召回 Top-K 工具。
        - 工具总数 <= k 或没有挂载 retriever 时，直接返回全量。
        - 否则委托给 retriever 做相似度计算。
        """
        if len(self.specs) <= k or self.retriever is None:
            return self.all()
        return self.retriever.search(query, k)

    def format_for_prompt(self, specs: Optional[List[ToolSpec]] = None) -> str:
        specs = specs if specs is not None else self.all()
        return "\n\n".join(s.format() for s in specs)


# ====================================================================
# 4. 内置工具规范
# ====================================================================

SEARCH_SPEC = ToolSpec(
    name="Search",
    description="一个网页搜索引擎，返回与查询最相关的网页摘要或直接答案。",
    category="search",
    parameters={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "自然语言搜索关键词"}
        },
        "required": ["query"],
    },
    when_to_use="需要回答关于时事、最新事件、人物、产品发布等知识库之外的事实性问题时。",
    when_not_to_use="纯数学计算、代码生成、对已知信息的整理 —— 这些场景不应使用本工具。",
    examples=[
        {"input": "英伟达最新的GPU型号是什么", "output": "Blackwell B200..."},
        {"input": "2025年诺贝尔物理学奖得主", "output": "..."},
    ],
    func=search,
)

CALCULATOR_SPEC = ToolSpec(
    name="Calculator",
    description="安全的数学表达式计算器，支持四则运算、幂、sqrt/sin/cos/log 等函数。",
    category="math",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "合法的 Python 数学表达式，例如 (1+2)*3 或 sqrt(16)",
            }
        },
        "required": ["expression"],
    },
    when_to_use="需要进行精确的数值计算时；尤其当问题中包含明确的数字与运算符。",
    when_not_to_use="问题涉及外部事实、最新信息或非数值推理时不应使用。",
    examples=[
        {"input": "(123 + 456) * 789 / 12", "output": "38068.25"},
        {"input": "sqrt(2) * pi", "output": "4.442882938..."},
    ],
    func=calculator,
)


# ====================================================================
# 5. 使用示例
# ====================================================================

if __name__ == "__main__":
    from tool_retriever import TfidfRetriever

    registry = ToolRegistry(retriever=TfidfRetriever())
    registry.register(SEARCH_SPEC)
    registry.register(CALCULATOR_SPEC)

    print("\n--- 全量工具描述 ---")
    print(registry.format_for_prompt())

    print("\n--- 针对'今天英伟达股价'的 Top-1 召回 ---")
    print(registry.format_for_prompt(registry.retrieve("今天英伟达股价", k=1)))

    print("\n--- 针对'计算 sqrt(2)*pi'的 Top-1 召回 ---")
    print(registry.format_for_prompt(registry.retrieve("计算 sqrt(2)*pi", k=1)))
