# my_custom_llm.py
"""
通过继承 HelloAgentsLLM 为框架新增一个模型供应商（Gemini）的支持。

设计要点（相比 7.2.1 的示例做了增强）：
- 7.2.1 的 MyLLM 直接重写整个 __init__，只有「显式传入 provider」时才生效，无法自动检测。
- 这里改为只重写父类初始化流程中的三个钩子方法，从而完整复用父类的
  初始化与自动检测流程，真正做到「自动检测该供应商的环境变量」：
    1. _auto_detect_provider : 优先识别 Gemini 的专属环境变量 / base_url
    2. _resolve_credentials  : 为 Gemini 解析 api_key 与 base_url
    3. _get_default_model    : 为 Gemini 提供默认模型

说明：deepseek 等供应商其实已内置于基类，因此这里选用基类尚未支持的
Gemini 作为「全新供应商」的演示。Gemini 提供 OpenAI 兼容端点，可直接接入。
"""

import os
from typing import Optional
from hello_agents import HelloAgentsLLM


class CustomLLM(HelloAgentsLLM):
    """一个自定义的 LLM 客户端，通过继承新增了对 Google Gemini 的支持。"""

    # Gemini 的 OpenAI 兼容端点与默认模型
    GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
    GEMINI_DEFAULT_MODEL = "gemini-1.5-flash"

    def _auto_detect_provider(self, api_key: Optional[str], base_url: Optional[str]) -> str:
        """重写自动检测逻辑：优先识别 Gemini，其余情况交还父类处理。"""
        # 1. 检查 Gemini 专属环境变量
        if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
            return "gemini"

        # 2. 根据 base_url 判断
        actual_base_url = base_url or os.getenv("LLM_BASE_URL")
        if actual_base_url and "generativelanguage.googleapis.com" in actual_base_url.lower():
            return "gemini"

        # 3. 不是 Gemini，则完全沿用父类的检测逻辑（openai / deepseek / qwen ...）
        return super()._auto_detect_provider(api_key, base_url)

    def _resolve_credentials(self, api_key: Optional[str], base_url: Optional[str]) -> tuple[str, str]:
        """重写凭证解析：为 Gemini 提供专属的 api_key 与 base_url 解析。"""
        if self.provider == "gemini":
            resolved_api_key = (
                api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("GOOGLE_API_KEY")
                or os.getenv("LLM_API_KEY")
            )
            resolved_base_url = base_url or os.getenv("LLM_BASE_URL") or self.GEMINI_BASE_URL

            if not resolved_api_key:
                raise ValueError(
                    "未找到 Gemini API 密钥，请设置 GEMINI_API_KEY（或 GOOGLE_API_KEY）环境变量。"
                )

            print("正在使用自定义的 Gemini Provider")
            return resolved_api_key, resolved_base_url

        # 其余供应商交还父类处理
        return super()._resolve_credentials(api_key, base_url)

    def _get_default_model(self) -> str:
        """重写默认模型：当用户未指定模型时，为 Gemini 提供默认模型。"""
        if self.provider == "gemini":
            return self.GEMINI_DEFAULT_MODEL
        return super()._get_default_model()
