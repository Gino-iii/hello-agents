# test_tot_agent.py
from dotenv import load_dotenv
from hello_agents.core.llm import HelloAgentsLLM
from my_tot_agent import TreeOfThoughtAgent

# 加载环境变量
load_dotenv()

# 创建LLM实例
llm = HelloAgentsLLM()

# 创建Tree-of-Thought Agent
# n_candidates: 每步生成几条候选思路；max_depth: 最多思考几步；beam_width: 每层保留几条最优路径
agent = TreeOfThoughtAgent(
    name="我的思维树助手",
    llm=llm,
    n_candidates=3,
    max_depth=3,
    beam_width=1,
)

# 测试一个需要试错与权衡的推理问题
question = (
    "使用数字 4、9、10、13，每个数字恰好用一次，"
    "通过加、减、乘、除四则运算，得到 24。请给出一个算式。"
)

result = agent.run(question)
print(f"\n最终结果: {result}")

# 查看对话历史
print(f"对话历史: {len(agent.get_history())} 条消息")
