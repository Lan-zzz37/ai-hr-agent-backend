"""
大模型 Agent 核心
基于 LangChain 枧构建对话式 Agent，支持工具调用和知识库检索
"""
from typing import List, Dict, Optional, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from backend.config import (
    LLM_API_KEY, LLM_API_BASE, LLM_MODEL_NAME,
    LLM_TEMPERATURE, LLM_MAX_TOKENS,
)
from backend.agent_core.prompts import SYSTEM_ROLE_PROMPT, get_prompt
from backend.database.chroma_client import chroma_manager


class SmartMfgAgent:
    """智能制造转型AI顾问 Agent"""

    def __init__(self, session_id: str = "default"):
        """
        Args:
            session_id: 会话ID，用于隔离不同用户的对话记忆
        """
        self.session_id = session_id
        self.llm = self._init_llm()
        self.memory = ConversationBufferWindowMemory(
            k=10,  # 保留最近10轮对话
            return_messages=True,
        )
        self._system_prompt = SYSTEM_ROLE_PROMPT

    def _init_llm(self) -> ChatOpenAI:
        """初始化大模型"""
        return ChatOpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_BASE,
            model=LLM_MODEL_NAME,
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    async def chat(self, user_message: str) -> str:
        """
        与 Agent 对话
        Args:
            user_message: 用户消息
        Returns:
            Agent 回复
        """
        # 构建消息列表
        messages = [SystemMessage(content=self._system_prompt)]

        # 加入历史记忆
        for msg in self.memory.load_memory_variables({}).get("history", []):
            messages.append(msg)

        # 加入当前用户消息
        messages.append(HumanMessage(content=user_message))

        # 调用大模型
        response = await self.llm.apredict_messages(messages)

        # 保存到记忆
        self.memory.save_context(
            {"input": user_message},
            {"output": response.content},
        )

        return response.content

    async def chat_stream(self, user_message: str) -> AsyncGenerator[str, None]:
        """
        流式对话
        Args:
            user_message: 用户消息
        Yields:
            Agent 回复的文本片段
        """
        messages = [SystemMessage(content=self._system_prompt)]

        for msg in self.memory.load_memory_variables({}).get("history", []):
            messages.append(msg)

        messages.append(HumanMessage(content=user_message))

        full_response = ""
        async for chunk in self.llm._stream_messages(messages):
            if chunk.content:
                full_response += chunk.content
                yield chunk.content

        self.memory.save_context(
            {"input": user_message},
            {"output": full_response},
        )

    async def analyze_match(
        self,
        employee_name: str,
        current_position: str,
        years_of_service: str,
        employee_skills: str,
        position_title: str,
        position_level: str,
        position_skills: str,
        responsibility: str,
        gap_analysis_data: str,
    ) -> str:
        """
        人岗匹配分析
        """
        prompt = get_prompt(
            "matching_analysis",
            employee_name=employee_name,
            current_position=current_position,
            years_of_service=years_of_service,
            employee_skills=employee_skills,
            position_title=position_title,
            position_level=position_level,
            position_skills=position_skills,
            responsibility=responsibility,
            gap_analysis_data=gap_analysis_data,
        )
        response = await self.llm.apredict(prompt)
        return response

    async def generate_skill_reshaping_plan(
        self,
        employee_name: str,
        current_skills: str,
        target_position: str,
        gap_skills: str,
    ) -> str:
        """
        生成技能重塑计划
        """
        prompt = get_prompt(
            "skill_reshaping",
            employee_name=employee_name,
            current_skills=current_skills,
            target_position=target_position,
            gap_skills=gap_skills,
        )
        response = await self.llm.apredict(prompt)
        return response

    async def knowledge_qa(self, user_question: str) -> str:
        """
        知识库问答（RAG）
        """
        # 检索相关知识
        try:
            collection = chroma_manager.get_collection("skill_knowledge")
            results = collection.query(
                query_texts=[user_question],
                n_results=3,
            )
            retrieved_context = "\n\n".join(results["documents"][0]) if results["documents"] else ""
        except Exception:
            retrieved_context = ""

        prompt = get_prompt(
            "knowledge_qa",
            retrieved_context=retrieved_context or "（知识库暂无相关内容）",
            user_question=user_question,
        )
        response = await self.llm.apredict(prompt)
        return response

    async def recommend_skill_direction(
        self,
        current_position: str,
        core_skills: str,
        experience_summary: str,
        interest_direction: str = "暂无",
    ) -> str:
        """
        技能转型方向推荐
        """
        prompt = get_prompt(
            "skill_recommendation",
            current_position=current_position,
            core_skills=core_skills,
            experience_summary=experience_summary,
            interest_direction=interest_direction,
        )
        response = await self.llm.apredict(prompt)
        return response

    def reset_memory(self):
        """重置对话记忆"""
        self.memory.clear()

    def set_system_prompt(self, custom_prompt: str):
        """自定义系统提示词"""
        self._system_prompt = custom_prompt


# ==================== Agent 会话管理 ====================
class AgentSessionManager:
    """管理多个 Agent 会话"""

    def __init__(self):
        self._sessions: Dict[str, SmartMfgAgent] = {}

    def get_agent(self, session_id: str) -> SmartMfgAgent:
        """获取或创建 Agent 会话"""
        if session_id not in self._sessions:
            self._sessions[session_id] = SmartMfgAgent(session_id=session_id)
        return self._sessions[session_id]

    def close_session(self, session_id: str):
        """关闭会话"""
        if session_id in self._sessions:
            del self._sessions[session_id]

    def list_sessions(self) -> List[str]:
        """列出所有活跃会话"""
        return list(self._sessions.keys())


# 全局会话管理器
agent_session_manager = AgentSessionManager()