"""
测试阿里云百炼 Embedding 连接
用法：cd backend && python test_embedding.py
"""
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
import config


def get_embedding(text: str) -> list:
    """调用阿里云百炼 Embedding API，返回向量列表"""
    client = OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_API_BASE,
    )
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL_NAME,
        input=text,
    )
    return response.data[0].embedding


if __name__ == "__main__":
    print("=" * 50)
    print("阿里云百炼 Embedding 连接测试")
    print("=" * 50)
    print(f"API Base:  {config.EMBEDDING_API_BASE}")
    print(f"Model:     {config.EMBEDDING_MODEL_NAME}")
    print("-" * 50)

    text = "我是一名钳工，想转行做机器人运维"
    print(f"输入类型:  {type(text)}")
    print(f"输入内容:  {repr(text)}")
    print("-" * 50)

    try:
        vector = get_embedding(text)
        print(f"向量维度:  {len(vector)}")
        print(f"向量前5维: {vector[:5]}")
        print()
        print("✅ 阿里云百炼 Embedding 连接成功！")
    except Exception as e:
        print(f"❌ 连接失败: {e}")