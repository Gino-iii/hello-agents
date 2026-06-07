# my_reflection_agent.py
"""扩展框架版 ReflectionAgent：增加"质量评分"提前终止机制。

在原有"执行 -> 反思 -> 优化"循环的基础上，每轮反思后新增一个评分环节：
让 LLM 对当前版本的输出按 0-10 打分，只有当分数 **低于阈值** 时才继续优化，
达到或超过阈值则提前终止，避免没有收益的多余迭代（同时节省 token 与时间）。
"""

import re
from typing import Optional, Dict
from hello_agents import ReflectionAgent, HelloAgentsLLM, Config, Message

# 评分提示词：要求模型输出结构化、可解析的分数
SCORE_PROMPT = """你是一位严格的质量评审专家。请根据原始任务，对下面这份回答的整体质量打分。

# 原始任务:
{task}

# 待评分的回答:
{content}

评分标准（满分 10 分）：综合考虑正确性、完整性、清晰度，以及是否真正满足任务要求。
请严格按照以下格式输出，先给分数再给一句话理由：
评分: <0到10之间的数字>
理由: <简要说明>
"""


class MyReflectionAgent(ReflectionAgent):
    """带质量评分提前终止机制的反思智能体。

    相比父类 ReflectionAgent，新增两个能力：
    1. 每轮反思后调用 LLM 对当前输出打分；
    2. 分数达到 score_threshold 即提前终止，不再进入优化步骤。
    """

    def __init__(
        self,
        name: str,
        llm: HelloAgentsLLM,
        system_prompt: Optional[str] = None,
        config: Optional[Config] = None,
        max_iterations: int = 3,
        custom_prompts: Optional[Dict[str, str]] = None,
        score_threshold: float = 8.0,
        score_prompt: Optional[str] = None,
    ):
        """
        Args:
            score_threshold: 质量分阈值（0-10），达到即提前终止
            score_prompt: 自定义评分提示词，需包含 {task} 与 {content} 占位符
        """
        super().__init__(name, llm, system_prompt, config, max_iterations, custom_prompts)
        self.score_threshold = score_threshold
        self.score_prompt = score_prompt or SCORE_PROMPT
        print(f"✅ {name} 初始化完成，最大迭代 {max_iterations} 轮，评分阈值 {score_threshold}")

    def run(self, input_text: str, **kwargs) -> str:
        """运行带评分提前终止的反思循环。"""
        print(f"\n🤖 {self.name} 开始处理任务: {input_text}")

        # 重置记忆（复用父类已创建的 Memory 实例）
        self.memory.records = []

        # 1. 初始执行
        print("\n--- 正在进行初始尝试 ---")
        initial_prompt = self.prompts["initial"].format(task=input_text)
        result = self._get_llm_response(initial_prompt, **kwargs)
        self.memory.add_record("execution", result)

        # 2. 迭代循环：反思 -> 评分 -> （按需）优化
        for i in range(self.max_iterations):
            print(f"\n--- 第 {i+1}/{self.max_iterations} 轮迭代 ---")
            last_result = self.memory.get_last_execution()

            # a. 反思
            print("\n-> 正在进行反思...")
            reflect_prompt = self.prompts["reflect"].format(
                task=input_text, content=last_result
            )
            feedback = self._get_llm_response(reflect_prompt, **kwargs)
            self.memory.add_record("reflection", feedback)

            # b. 文本信号：反思直接判定无需改进
            if "无需改进" in feedback or "no need for improvement" in feedback.lower():
                print("\n✅ 反思认为结果已无需改进，提前终止。")
                break

            # c. 质量评分：让 LLM 对当前输出打分
            print("\n-> 正在进行质量评分...")
            score = self._score_output(input_text, last_result, **kwargs)
            self.memory.add_record("score", f"{score}")
            print(f"📊 当前输出评分: {score}/10（阈值 {self.score_threshold}）")

            # d. 达到阈值则提前终止，不再优化
            if score >= self.score_threshold:
                print(f"\n✅ 评分已达阈值，质量满足要求，提前终止。")
                break

            # e. 分数不足，根据反馈进行优化
            print("\n-> 正在进行优化...")
            refine_prompt = self.prompts["refine"].format(
                task=input_text,
                last_attempt=last_result,
                feedback=feedback,
            )
            refined = self._get_llm_response(refine_prompt, **kwargs)
            self.memory.add_record("execution", refined)

        final_result = self.memory.get_last_execution()
        print(f"\n--- 任务完成 ---\n最终结果:\n{final_result}")

        # 保存到历史记录
        self.add_message(Message(input_text, "user"))
        self.add_message(Message(final_result, "assistant"))
        return final_result

    def _score_output(self, task: str, content: str, **kwargs) -> float:
        """让 LLM 对当前输出打分，返回 0-10 的浮点分数。"""
        prompt = self.score_prompt.format(task=task, content=content)
        response = self._get_llm_response(prompt, **kwargs)
        return self._parse_score(response)

    @staticmethod
    def _parse_score(text: str) -> float:
        """从 LLM 回复中稳健地解析分数。

        解析失败时返回 0.0（视为不达标，继续优化），避免因解析异常误判为"已达标"而提前终止。
        """
        if not text:
            return 0.0
        # 优先匹配"评分: X"形式
        match = re.search(r"评分[:：]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            # 退化策略：取文本中出现的第一个数字
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        if not match:
            return 0.0
        score = float(match.group(1))
        # 约束到 [0, 10]，防止异常分数破坏阈值判断
        return max(0.0, min(10.0, score))


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    llm = HelloAgentsLLM()

    agent = MyReflectionAgent(
        name="带评分的反思助手",
        llm=llm,
        max_iterations=3,
        score_threshold=8.0,
    )

    result = agent.run("写一篇关于人工智能发展历程的简短文章")
    print(f"\n最终结果:\n{result}")
