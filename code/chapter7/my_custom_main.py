# my_custom_main.py
"""
演示如何使用通过继承扩展出的 Gemini 供应商支持。

用法一：显式指定 provider
    llm = CustomLLM(provider="gemini")

用法二：自动检测（推荐）
    只要在 .env 中配置了 GEMINI_API_KEY（或 GOOGLE_API_KEY），
    无需传入 provider，框架会自动检测为 'gemini'：
        GEMINI_API_KEY="your-gemini-api-key"
"""

from dotenv import load_dotenv
from my_custom_llm import CustomLLM

load_dotenv()

# --- 用法二：自动检测 ---
# 由于重写了 _auto_detect_provider，配置 GEMINI_API_KEY 后即可自动识别
llm = CustomLLM()
print(f"检测到的 provider: {llm.provider}")
print(f"使用的模型: {llm.model}")
print(f"服务地址: {llm.base_url}")

messages = [{"role": "user", "content": "你好，请用一句话介绍一下你自己。"}]

response_stream = llm.think(messages)

print("\nGemini Response:")
for chunk in response_stream:
    # think 方法已在内部打印过流式内容，这里 pass 即可
    pass
