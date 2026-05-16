from dotenv import load_dotenv
# 加载 .env 文件中的环境变量
load_dotenv()

import os
from serpapi import SerpApiClient
from typing import Dict, Any

def search(query: str) -> str:
    """
    一个基于SerpApi的实战网页搜索引擎工具。
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
            "gl": "cn",  # 国家代码
            "hl": "zh-cn", # 语言代码
        }
        
        client = SerpApiClient(params)
        results = client.get_dict()
        
        # 智能解析：优先寻找最直接的答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            # 如果没有直接答案，则返回前三个有机结果的摘要
            snippets = [
                f"[{i+1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        
        return f"对不起，没有找到关于 '{query}' 的信息。"

    except Exception as e:
        return f"搜索时发生错误: {e}"
    
import math
import operator

def calculator(expression: str) -> str:
    """
    一个安全的数学计算器工具。
    支持加减乘除、括号、幂运算、以及常用数学函数（sqrt、sin、cos 等）。
    """
    print(f"🧮 正在执行计算: {expression}")
    try:
        # 允许使用的安全命名空间：仅开放 math 模块中的函数和常量
        safe_globals = {
            "__builtins__": {},  # 禁止所有内置函数，防止代码注入
        }
        safe_locals = {
            # 基础数学函数
            "sqrt": math.sqrt,
            "pow": math.pow,
            "abs": abs,
            "round": round,
            "log": math.log,
            "log10": math.log10,
            "log2": math.log2,
            "exp": math.exp,
            # 三角函数
            "sin": math.sin,
            "cos": math.cos,
            "tan": math.tan,
            "asin": math.asin,
            "acos": math.acos,
            "atan": math.atan,
            # 常量
            "pi": math.pi,
            "e": math.e,
            # 取整
            "floor": math.floor,
            "ceil": math.ceil,
        }

        # 对表达式做基本安全检查，拒绝包含危险关键字的输入
        forbidden_keywords = ["import", "exec", "eval", "open", "os", "sys", "__"]
        for keyword in forbidden_keywords:
            if keyword in expression.lower():
                return f"错误：表达式包含不允许的关键字 '{keyword}'。"

        result = eval(expression, safe_globals, safe_locals)  # noqa: S307

        # 格式化输出：整数结果不显示小数点
        if isinstance(result, float) and result.is_integer():
            return f"计算结果：{expression} = {int(result)}"
        return f"计算结果：{expression} = {result}"

    except ZeroDivisionError:
        return "错误：除数不能为零。"
    except SyntaxError:
        return f"错误：表达式语法有误，请检查 '{expression}'。"
    except Exception as e:
        return f"计算时发生错误: {e}"


from typing import Dict, Any

class ToolExecutor:
    """
    一个工具执行器，负责管理和执行工具。
    """
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        """
        向工具箱中注册一个新工具。
        """
        if name in self.tools:
            print(f"警告：工具 '{name}' 已存在，将被覆盖。")
        
        self.tools[name] = {"description": description, "func": func}
        print(f"工具 '{name}' 已注册。")

    def getTool(self, name: str) -> callable:
        """
        根据名称获取一个工具的执行函数。
        """
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        """
        获取所有可用工具的格式化描述字符串。
        """
        return "\n".join([
            f"- {name}: {info['description']}" 
            for name, info in self.tools.items()
        ])


# --- 工具初始化与使用示例 ---
if __name__ == '__main__':
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)
    
    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误：未找到名为 '{tool_name}' 的工具。")
