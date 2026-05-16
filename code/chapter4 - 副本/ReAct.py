import re
from llm_client import HelloAgentsLLM
from tools import ToolExecutor, search

# (此处省略 REACT_PROMPT_TEMPLATE 的定义)
REACT_PROMPT_TEMPLATE = """
请注意，你是一个有能力调用外部工具的智能助手。

可用工具如下：
{tools}

请严格按照以下格式进行回应：

Thought: 你的思考过程，用于分析问题、拆解任务和规划下一步行动。
Action: 你决定采取的行动，必须是以下格式之一：
- `{{tool_name}}[{{tool_input}}]`：调用一个可用工具。
- `Finish[最终答案]`：当你认为已经获得最终答案时。
- 当你收集到足够的信息，能够回答用户的最终问题时，你必须在`Action:`字段后使用 `Finish[最终答案]` 来输出最终答案。


现在，请开始解决以下问题：
Question: {question}
History: {history}
"""

TOOL_FAILURE_HINT_TEMPLATE = """
⚠️ 注意：你已经连续 {count} 次工具调用失败（失败原因：{reason}）。
请重新审视上方的【可用工具列表】，确认：
1. 工具名称是否拼写正确？只能使用列表中存在的工具名。
2. 工具输入格式是否符合要求？
3. 当前任务是否适合换一个工具来完成？
请调整你的策略，不要重复同样的错误。
"""


class ReActAgent:
    def __init__(
        self,
        llm_client: HelloAgentsLLM,
        tool_executor: ToolExecutor,
        max_steps: int = 5,
        max_failures: int = 3,
    ):
        self.llm_client = llm_client
        self.tool_executor = tool_executor
        self.max_steps = max_steps
        self.max_failures = max_failures
        self.history = []

    def run(self, question: str):
        self.history = []
        current_step = 0
        consecutive_failures = 0  # 连续失败计数器
        failure_reasons = []  # 记录每次失败的原因

        while current_step < self.max_steps:
            current_step += 1
            print(f"\n--- 第 {current_step} 步 ---")

            # 如果连续失败次数达到阈值，在提示词中注入纠错引导
            if consecutive_failures >= self.max_failures:
                print(f"⚠️  已连续失败 {consecutive_failures} 次，向模型注入纠错提示...")
                failure_hint = TOOL_FAILURE_HINT_TEMPLATE.format(
                    count=consecutive_failures, reason="\n".join(failure_reasons)
                )
            else:
                failure_hint = ""  # 正常情况下不注入额外提示

            tools_desc = self.tool_executor.getAvailableTools()
            history_str = "\n".join(self.history)
            prompt = REACT_PROMPT_TEMPLATE.format(
                tools=tools_desc,
                question=question,
                history=history_str,
                failure_hint=failure_hint,
            )

            messages = [{"role": "user", "content": prompt}]
            response_text = self.llm_client.think(messages=messages)
            if not response_text:
                print("错误：LLM未能返回有效响应。")
                break
            # 解析输出
            thought, action = self._parse_output(response_text)

            if thought:
                print(f"🤔 思考: {thought}")
            if not action:
                print("警告：未能解析出有效的Action，流程终止。")
                break

            if action.startswith("Finish"):
                # 如果是Finish指令，提取最终答案并结束
                final_answer = self._parse_action_input(action)
                print(f"🎉 最终答案: {final_answer}")
                return final_answer

            tool_name, tool_input = self._parse_action(action)
            # 失败情况1
            if not tool_name or not tool_input:
                consecutive_failures += 1
                failure_reasons.append(
                    f"第{consecutive_failures}次：Action 格式无法解析：'{action}'，正确格式为 ToolName[input]"
                )
                print(f"❌ 格式错误（连续失败 {consecutive_failures} 次）")
                self.history.append(f"Action: {action}")
                self.history.append(
                    "Observation: 错误：Action格式无效，请使用 ToolName[input] 格式。"
                )
                continue

            print(f"🎬 行动: {tool_name}[{tool_input}]")
            tool_function = self.tool_executor.getTool(tool_name)
            # 失败情况2 工具名不存在
            if not tool_function:
                consecutive_failures += 1
                failure_reasons.append(
                    f"第{consecutive_failures}次：工具名 '{tool_name}' 不存在，可用工具为：{list(self.tool_executor.tools.keys())}"
                )
                print(f"❌ 工具未找到（连续失败 {consecutive_failures} 次）")
                self.history.append(f"Action: {action}")
                self.history.append(
                    f"Observation: 错误：未找到名为 '{tool_name}' 的工具。请从可用工具列表中选择正确的工具名。"
                )
                continue
            # 执行工具
            observation = tool_function(tool_input)
            print(f"👀 观察: {observation}")
            # 失败情况3 工具执行返回了错误信息
            if isinstance(observation, str) and observation.startswith("错误："):
                consecutive_failures += 1
                failure_reasons.append(
                    f"第{consecutive_failures}次：工具 '{tool_name}' 执行出错：{observation}"
                )
                print(f"❌ 工具执行失败（连续失败 {consecutive_failures} 次）")
                self.history.append(f"Action: {action}")
                self.history.append(
                    f"Observation: 错误：工具 '{tool_name}' 执行出错：{observation}"
                )
            else:
                # 工具调用成功，重置失败计数和原因列表
                consecutive_failures = 0
                failure_reasons = []

            self.history.append(f"Action: {action}")
            self.history.append(f"Observation: {observation}")

        print("已达到最大步数，流程终止。")
        return None

    def _parse_output(self, text: str):
        # Thought: 匹配到 Action: 或文本末尾
        thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|$)", text, re.DOTALL)
        # Action: 匹配到文本末尾
        action_match = re.search(r"Action:\s*(.*?)$", text, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else None
        action = action_match.group(1).strip() if action_match else None
        return thought, action

    def _parse_action(self, action_text: str):
        match = re.match(r"(\w+)\[(.*)\]", action_text, re.DOTALL)
        return (match.group(1), match.group(2)) if match else (None, None)

    def _parse_action_input(self, action_text: str):
        match = re.match(r"\w+\[(.*)\]", action_text, re.DOTALL)
        return match.group(1) if match else ""


if __name__ == "__main__":
    llm = HelloAgentsLLM()
    tool_executor = ToolExecutor()
    search_desc = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    tool_executor.registerTool("Search", search_desc, search)
    agent = ReActAgent(llm_client=llm, tool_executor=tool_executor)
    question = "华为最新的手机是哪一款？它的主要卖点是什么？"
    agent.run(question)
