"""
测试阿里云百炼 LLM 连接
用法：cd backend && python test_llm.py
"""
from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import config

print("=" * 50)
print("阿里云百炼 LLM 连接测试")
print("=" * 50)
print(f"API Base:  {config.LLM_API_BASE}")
print(f"Model:     {config.LLM_MODEL_NAME}")
print("-" * 50)

try:
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0,
        max_tokens=50,
    )
    response = llm.invoke([HumanMessage(content="请回复：连接成功")])
    print(f"LLM 响应: {response.content}")
    print()
    print("✅ 阿里云百炼 LLM 连接成功！")
except Exception as e:
    print(f"❌ 连接失败: {e}")