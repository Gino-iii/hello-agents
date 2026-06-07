"""Tree-of-Thought Agent 实现 - 思维树搜索的智能体

核心思想（Tree-of-Thought, ToT）：
    把推理过程建模成一棵"思维树"。与 ReAct / Plan-and-Solve 那种"单链条"的
    线性推理不同，ToT 在每一步都会：
        1. 扩展(Propose)：从当前最优的思考节点出发，生成多个不同的候选思路；
        2. 评估(Evaluate)：用 LLM 作为价值函数，给每条候选路径打分；
        3. 选择(Select)：保留得分最高的若干条路径（Beam Search）继续向下探索。
    通过"生成多条路径 -> 评估 -> 择优继续"的循环，ToT 能够进行带回溯、带剪枝
    的探索式推理，特别适合解谜、数学推导、策略规划等需要试错与权衡的任务。

参考: Yao et al., "Tree of Thoughts: Deliberate Problem Solving with LLMs" (2023)
"""

import re
import ast
from typing import Optional, List, Dict, Tuple

from hello_agents.core.agent import Agent
from hello_agents.core.llm import HelloAgentsLLM
from hello_agents.core.config import Config
from hello_agents.core.message import Message


# ============================ 默认提示词模板 ============================

# 扩展器提示词：基于已有思考路径，生成多个不同的下一步思路
DEFAULT_PROPOSE_PROMPT = """你是一位富有创造力的推理专家。我们正在用"思维树"的方式逐步解决一个问题。
请基于"已有的思考路径"，为"下一步"提出 {n} 个**互不相同**且具有探索价值的思路。

# 要解决的问题:
{question}

# 已有的思考路径:
{history}

# 要求:
1. 每个思路都是在已有路径之上向前推进的"一步"，要具体、可执行，而不是空泛的口号。
2. {n} 个思路应当尽可能体现**不同的方向或策略**，便于后续比较择优。
3. 如果某个思路已经能够直接得出最终答案，请在该思路中明确写出答案。

请严格按照以下格式输出一个 Python 列表（仅输出列表，不要任何额外说明）:
```python
["思路1", "思路2", "思路3"]
```
"""

# 评估器提示词：作为价值函数，对一条思考路径打分
DEFAULT_EVALUATE_PROMPT = """你是一位严谨的推理评审专家。请评估下面这条"思考路径"对于解决目标问题的**前景与质量**。

# 要解决的问题:
{question}

# 待评估的思考路径:
{history}

# 评估标准:
- 这条路径的推理是否正确、自洽？
- 它是否在朝着正确答案稳步推进？是否更接近最终解？
- 是否存在明显的逻辑错误、死胡同或无效绕路？

请严格按照以下格式输出（不要输出多余内容）:
评分: <0到10之间的整数，10表示几乎肯定能得到正确答案，0表示完全错误或走入死胡同>
是否解决: <是 / 否，仅当这条路径已经给出了完整且正确的最终答案时才填"是">
理由: <一句话说明你的评分依据>
"""

# 合成器提示词：基于最优路径，输出最终答案
DEFAULT_FINALIZE_PROMPT = """你是一位总结专家。下面是经过"思维树"搜索后筛选出的、最优的一条完整思考路径。
请基于这条路径，给出对原始问题清晰、准确的最终答案。

# 原始问题:
{question}

# 最优思考路径:
{history}

请直接输出最终答案（可附简要说明），不要重复罗列上面的推理步骤:
"""


# ============================ 思维树节点 ============================

class ThoughtNode:
    """思维树中的一个节点，代表"一步思考"。

    多个节点通过 parent 指针串成一条从根到叶的"思考路径"。
    """

    def __init__(self, thought: Optional[str], parent: Optional["ThoughtNode"] = None, depth: int = 0):
        self.thought = thought          # 当前这一步的思考内容（根节点为 None）
        self.parent = parent            # 父节点
        self.depth = depth              # 节点深度（根节点为 0）
        self.score: float = 0.0         # 评估器给出的得分
        self.solved: bool = False       # 是否已被判定为"完整正确答案"
        self.children: List["ThoughtNode"] = []  # 子节点

    @property
    def path(self) -> List[str]:
        """返回从根到当前节点的思考序列（不含空的根节点）。"""
        steps: List[str] = []
        node: Optional[ThoughtNode] = self
        while node is not None:
            if node.thought is not None:
                steps.append(node.thought)
            node = node.parent
        return list(reversed(steps))

    def path_text(self) -> str:
        """把思考路径渲染成可读文本，用于填充提示词。"""
        steps = self.path
        if not steps:
            return "（暂无思考步骤，这是第一步）"
        return "\n".join(f"步骤{i}: {s}" for i, s in enumerate(steps, 1))


# ============================ 扩展器 ============================

class ThoughtGenerator:
    """扩展器 - 从一个节点出发，生成多个候选的下一步思路。"""

    def __init__(self, llm: HelloAgentsLLM, prompt_template: Optional[str] = None, temperature: float = 0.9):
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_PROPOSE_PROMPT
        # 生成阶段用较高温度，鼓励路径的多样性
        self.temperature = temperature

    def generate(self, question: str, node: ThoughtNode, n_candidates: int, **kwargs) -> List[str]:
        prompt = self.prompt_template.format(
            question=question,
            history=node.path_text(),
            n=n_candidates,
        )
        messages = [{"role": "user", "content": prompt}]
        kwargs.setdefault("temperature", self.temperature)
        response = self.llm.invoke(messages, **kwargs) or ""
        return self._parse_candidates(response, n_candidates)

    def _parse_candidates(self, response: str, n_candidates: int) -> List[str]:
        """从 LLM 响应中解析出候选思路列表，带多重兜底。"""
        # 优先解析 ```python [...] ``` 代码块中的列表
        try:
            block = response.split("```python")[1].split("```")[0].strip()
            parsed = ast.literal_eval(block)
            if isinstance(parsed, list):
                items = [str(x).strip() for x in parsed if str(x).strip()]
                if items:
                    return items[:n_candidates]
        except (ValueError, SyntaxError, IndexError):
            pass

        # 兜底：尝试解析任意中括号包裹的列表
        try:
            match = re.search(r"\[.*\]", response, re.DOTALL)
            if match:
                parsed = ast.literal_eval(match.group(0))
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
                    if items:
                        return items[:n_candidates]
        except (ValueError, SyntaxError):
            pass

        # 再兜底：按行解析，去掉行首的序号/项目符号
        lines = []
        for line in response.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^[\-\*\d\.\)、：:]+\s*", "", line)
            if line:
                lines.append(line)
        return lines[:n_candidates]


# ============================ 评估器 ============================

class ThoughtEvaluator:
    """评估器 - 作为价值函数，对一条思考路径进行打分。"""

    def __init__(self, llm: HelloAgentsLLM, prompt_template: Optional[str] = None, temperature: float = 0.2):
        self.llm = llm
        self.prompt_template = prompt_template or DEFAULT_EVALUATE_PROMPT
        # 评估阶段用较低温度，保证打分尽量稳定一致
        self.temperature = temperature

    def evaluate(self, question: str, node: ThoughtNode, **kwargs) -> Tuple[float, bool, str]:
        """返回 (得分0-10, 是否解决, 理由)。"""
        prompt = self.prompt_template.format(question=question, history=node.path_text())
        messages = [{"role": "user", "content": prompt}]
        kwargs.setdefault("temperature", self.temperature)
        response = self.llm.invoke(messages, **kwargs) or ""
        return self._parse_evaluation(response)

    def _parse_evaluation(self, response: str) -> Tuple[float, bool, str]:
        # 解析评分
        score = 5.0
        score_match = re.search(r"评分[:：]?\s*(\d+(?:\.\d+)?)", response)
        if score_match:
            try:
                score = max(0.0, min(10.0, float(score_match.group(1))))
            except ValueError:
                pass

        # 解析是否解决
        solved = False
        solved_match = re.search(r"是否解决[:：]?\s*(是|否|yes|no)", response, re.IGNORECASE)
        if solved_match:
            solved = solved_match.group(1).lower() in ("是", "yes")

        # 解析理由
        reason = ""
        reason_match = re.search(r"理由[:：]?\s*(.+)", response, re.DOTALL)
        if reason_match:
            reason = reason_match.group(1).strip().splitlines()[0]

        return score, solved, reason


# ============================ Tree-of-Thought Agent ============================

class TreeOfThoughtAgent(Agent):
    """Tree-of-Thought Agent - 基于思维树搜索的智能体。

    工作流程（Beam Search 版本的 ToT）：
        1. 从空的根节点开始，维护一个"束(beam)"——当前最优的若干条路径；
        2. 每一层，对 beam 中的每个节点调用扩展器，生成 n_candidates 个候选子节点；
        3. 用评估器给每个子节点打分；
        4. 在本层所有子节点中保留得分最高的 beam_width 个，作为新的 beam；
        5. 若某条路径被判定为"已解决"且得分达标，则提前停止；
        6. 到达最大深度或提前停止后，用合成器基于最优路径输出最终答案。

    可调参数：
        n_candidates: 每个节点每步扩展出的候选思路数量（树的分支因子）。
        max_depth:    思维树的最大深度（最多推理多少步）。
        beam_width:   每层保留的最优路径数。=1 即"生成多路 -> 选最优 -> 继续"的贪心搜索；
                      >1 则保留多条候选路径并行探索，更接近完整的束搜索。
        solved_threshold: 提前停止所需的最低得分（配合"是否解决=是"）。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        n_candidates: int = 3,
        max_depth: int = 3,
        beam_width: int = 1,
        solved_threshold: float = 8.0,
        custom_prompts: Optional[Dict[str, str]] = None,
    ):
        super().__init__(name, llm, system_prompt, config)
        self.n_candidates = max(1, n_candidates)
        self.max_depth = max(1, max_depth)
        self.beam_width = max(1, beam_width)
        self.solved_threshold = solved_threshold

        prompts = custom_prompts or {}
        self.generator = ThoughtGenerator(self.llm, prompts.get("propose"))
        self.evaluator = ThoughtEvaluator(self.llm, prompts.get("evaluate"))
        self.finalize_prompt = prompts.get("finalize") or DEFAULT_FINALIZE_PROMPT

        print(
            f"✅ {name} 初始化完成 "
            f"(分支={self.n_candidates}, 深度={self.max_depth}, 束宽={self.beam_width})"
        )

    def run(self, input_text: str, **kwargs) -> str:
        """运行 Tree-of-Thought Agent。"""
        print(f"\n🌳 {self.name} 开始用思维树求解: {input_text}")

        root = ThoughtNode(thought=None, depth=0)
        beam: List[ThoughtNode] = [root]   # 当前层保留的最优节点集合
        best_node: ThoughtNode = root      # 全局最优节点

        for depth in range(1, self.max_depth + 1):
            print(f"\n{'='*48}\n🔎 第 {depth} 层探索（从 {len(beam)} 个节点扩展）\n{'='*48}")

            children: List[ThoughtNode] = []

            # 1. 扩展 + 评估：对 beam 中每个节点生成并打分其候选子节点
            for parent in beam:
                candidates = self.generator.generate(
                    input_text, parent, self.n_candidates, **kwargs
                )
                if not candidates:
                    continue

                for idx, thought in enumerate(candidates, 1):
                    child = ThoughtNode(thought=thought, parent=parent, depth=depth)
                    score, solved, reason = self.evaluator.evaluate(input_text, child, **kwargs)
                    child.score = score
                    child.solved = solved
                    parent.children.append(child)
                    children.append(child)

                    flag = " ✅已解决" if solved else ""
                    print(f"  [候选{idx}] 评分={score:.1f}{flag} | {thought}")
                    if reason:
                        print(f"          ↳ 评审: {reason}")

            if not children:
                print("⚠️ 本层未能生成任何有效候选，提前结束探索。")
                break

            # 2. 选择：保留本层得分最高的 beam_width 个节点
            children.sort(key=lambda n: n.score, reverse=True)
            beam = children[: self.beam_width]
            best_node = beam[0]
            print(f"\n🏆 本层最优路径 (评分={best_node.score:.1f}): {best_node.thought}")

            # 3. 提前停止：已找到高质量的完整解
            solved_node = next(
                (n for n in children if n.solved and n.score >= self.solved_threshold),
                None,
            )
            if solved_node is not None:
                best_node = solved_node
                print(f"\n🎯 第 {depth} 层已找到高置信度的解，提前结束探索。")
                break

        # 4. 合成最终答案
        final_answer = self._finalize(input_text, best_node, **kwargs)

        print(f"\n{'='*48}\n✅ 思维树搜索完成\n{'='*48}")
        print(f"📜 最优思考路径:\n{best_node.path_text()}")
        print(f"\n💡 最终答案: {final_answer}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_answer, "assistant"))

        return final_answer

    def _finalize(self, question: str, node: ThoughtNode, **kwargs) -> str:
        """基于最优路径合成最终答案。"""
        prompt = self.finalize_prompt.format(question=question, history=node.path_text())
        messages = [{"role": "user", "content": prompt}]
        return (self.llm.invoke(messages, **kwargs) or "").strip()
