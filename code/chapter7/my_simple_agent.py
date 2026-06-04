# my_simple_agent.py
from typing import Optional, Iterator
from hello_agents import SimpleAgent, HelloAgentsLLM, Config, Message
import re


class MySimpleAgent(SimpleAgent):
     """
    重写的简单对话Agent
    展示如何基于框架基类构建自定义Agent
    """
    def __init__(
        self,
        name:str,
        llm:HelloAgentsLLM,
        system_prompt:Optional[str]=None,
        config:Optional[Config]=None,
        tool_registry:Optional['ToolRegistry']=None,
        enable_tool_calling:bool=True
    ):
        super().__init__(name, llm, system_prompt, config)
        self.tool_registry = tool_registry
        self.enable_tool_calling = enable_tool_calling and tool_registry is not None
        print(f{name}初始化完成，工具调用: {'启用' if self.enable_tool_calling else '禁用'})


    def _get_enhanced_system_prompt(self)->str:
        """构建增强的系统提示词，包含工具信息"""
        base_prompt=self.system_prompt or "你是一个有用的AI助手。"

        if not self.enable_tool_calling or not self.tool_registry:
            return base_prompt
        
        tool_description=self.tool_registry.get_tools_description()
        if not tool_description or tool_description == "暂无可用工具":
            return base_prompt
        
        tools_section = "\n\n## 可用工具\n"
        tools_section += "你可以使用以下工具来帮助回答问题：\n"
        tools_section += tool_description + "\n"
        tools_section += "\n## 工具调用格式\n"
        tools_section += "当需要使用工具时，请使用以下格式：\n"
        tools_section += "`[TOOL_CALL:{tool_name}:{parameters}]`\n"
        tools_section += "例如：`[TOOL_CALL:search:Python编程]` 或 `[TOOL_CALL:memory:recall=用户信息]`\n\n"
        tools_section += "工具调用结果会自动插入到对话中，然后你可以基于结果继续回答。\n"
        return base_prompt + tools_section

    def run(self,input_text:str,max_tool_iterations:int=3,**kwargs)->str:
        """
        重写的运行方法 - 实现简单对话逻辑，支持可选工具调用
        """
        print(f"🤖 {self.name} 正在处理: {input_text}")
        messages = []

        # 添加系统信息
        enhanced_system_prompt=self._get_enhanced_system_prompt()
        messages.append({"role": "system", "content": enhanced_system_prompt})

        # 添加历史信息
        for msg in self._history:
            messages.append({"role": msg.role, "content": msg.content})

        # 添加当前用户信息
        messages.append({"role": "user", "content": input_text})

        # 如果没有启用工具调用，使用简单对话逻辑
        if not self.enable_tool_calling:
            response = self.llm.invoke(messages, **kwargs)
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(response, "assistant"))
            print(f"✅ {self.name} 响应完成")
            return response
        
        # 支持多轮工具调用的逻辑
        return self._run_with_tools(messages, input_text, max_tool_iterations, **kwargs)

    def _parse_tool_parameters(self,tool_name:str,parameters:str)->dict:
        # 智能解析工具参数
        param_dict={}

        if "=" in parameters:
            if "," in parameters"
            # 多个参数：action=search ，query=Python,limit=3
                params=parameters.split(",")
                
                for param in params:
                    if "=" in param:
                        key, value=param.split("=", 1)
                        param_dict[key.strip()]=value.strip()
            else:
                # 单个参数：query=Python
                key, value=parameters.split("=", 1)
                param_dict[key.strip()]=value.strip()
        else:
            # 简单参数列表
            params=[p.strip() for p in parameters.split(",") if p.strip()]
            if len(params) == 1:
                param_dict["query"] = params[0]
            else:
                for i, param in enumerate(params):
                    param_dict[f"arg{i+1}"] = param
        return param_dict

        def _execute_tool_call(self,tool_name :str,parameters:str)->str:
            # 执行工具调用
            if not self.tool_registry:
                return f"❌ 错误:未配置工具注册表"

            try :
                # 智能参数解析
                if tool_name="calculator":
                 # 计算器工具直接传入表达式
                    result=self.tool_registry.execute_tool(tool_name,parameters)
                else:
                # 其他工具使用智能参数解析
                param_dict=self._parse_tool_parameters(tool_name,parameters)
                tool=self.tool_registry.get_tool(tool_name)
                if not tool:
                    return f"❌ 错误:未找到工具 '{tool_name}'"
                result=tool.run(param_dict)

                return f"🔧 工具 {tool_name} 执行结果：\n{result}"

                except Exception as e:
                    return f"❌ 工具调用失败：{str(e)}"
                    
        def stream_run(self,input_text:str,**kwargs)->Iterator[str]:
            """
            自定义的流式运行方法
            """
            print(f"🌊 {self.name} 开始流式处理: {input_text}")

            messages=[]

            if self.system_prompt:
                messages.append({"role": "system", "content": self.system_prompt})
            
            for msg in self._history:
                messages.append({"role": msg.role, "content": msg.content})

            messages.append({"role": "user", "content": input_text})

            full_response=""
            print("📝 实时响应: ", end="")
            for chunk in self.llm.stream_invoke(messages, **kwargs):
                full_response+=chunk
                print(chunk, end="", flush=True)
                yield chunk

            print()

            # 保存完整的对话到历史记录
            self.add_message(Message(input_text, "user"))
            self.add_message(Message(full_response, "assistant"))
            print(f"✅ {self.name} 流式响应完成")
            
        def add_tool(self,tool)->None:
            """添加工具到Agent（便利方法）"""
            if not self.tool_registry:
                from hello_agents import ToolRegistry
                self.tool_registry = ToolRegistry()
                self.enable_tool_calling = True
            self.tool_registry.register_tool(tool)
            print(f"🔧 工具 '{tool.name}' 已添加")

        def has_tools(self)->bool:
            """检查是否有可用工具"""
            return self.enable_tool_calling and self.tool_registry is not None
        
        def remove_tool(self,tool_name:str)->bool:
            """移除工具（便利方法）"""
            if self.tool_registry:
                self.tool_registry.unregister(tool_name)
                return True
            return False
   