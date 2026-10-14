"""
向量化模块
将文本数据（简历、岗位描述等）转化为向量并存储到 ChromaDB
"""
import uuid
from typing import List, Dict, Optional
from langchain.embeddings import Embeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from backend.config import (
    EMBEDDING_PROVIDER, EMBEDDING_API_KEY,
    EMBEDDING_API_BASE, EMBEDDING_MODEL_NAME,
)
from backend.database.chroma_client import chroma_manager


class DataVectorizer:
    """数据向量化处理器"""

    def __init__(self):
        self.embedding = self._init_embedding()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    def _init_embedding(self) -> Embeddings:
        """初始化 Embedding 模型"""
        if EMBEDDING_PROVIDER == "openai":
            return OpenAIEmbeddings(
                api_key=EMBEDDING_API_KEY,
                base_url=EMBEDDING_API_BASE,
                model=EMBEDDING_MODEL_NAME,
            )
        # 可扩展其他 Embedding 提供商
        return OpenAIEmbeddings(
            api_key=EMBEDDING_API_KEY,
            base_url=EMBEDDING_API_BASE,
            model=EMBEDDING_MODEL_NAME,
        )

    def vectorize_employee_resume(
        self,
        employee_id: str,
        resume_text: str,
        skills: List[Dict],
    ) -> str:
        """
        将员工简历和技能信息向量化并存入 ChromaDB
        Args:
            employee_id: 哗工ID
            resume_text: 简历文本
            skills: 抛能列表
        Returns:
            ChromaDB 中的文档ID
        """
        # 构结构化文本
        skill_descriptions = []
        for skill in skills:
            skill_descriptions.append(
                f"- {skill['name']}：熟练度{skill.get('proficiency_level', 0):.0%}，"
                f"经验{skill.get('years_of_experience', 0)}年"
            )

        combined_text = f"员工简历：\n{resume_text}\n\n技能列表：\n" + "\n".join(skill_descriptions)

        # 分块
        chunks = self.text_splitter.split_text(combined_text)

        # 生成ID
        doc_id = f"emp_{employee_id}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks)]
        metadatas = [
            {
                "employee_id": employee_id,
                "chunk_index": i,
                "type": "employee_profile",
            }
            for i in range(len(chunks)]
        ]

        # 存入 ChromaDB
        collection = chroma_manager.get_collection("employee_profiles")
        # 先删除旧数据
        try:
            collection.delete(where={"employee_id": employee_id})
        except Exception:
            pass

        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadatas,
        )

        return doc_id

    def vectorize_position_description(
        self,
        position_id: str,
        title: str,
        description: str,
        skills: List[Dict],
    ) -> str:
        """
        将岗位描述向量化并存入 ChromaDB
        Args:
            position_id: 岗位ID
            title: 岗位名称
            description: 嗷位描述
            skills: 抠能要求列表
        Returns:
            ChromaDB 中的文档ID
        """
        skill_requirements = []
        for skill in skills:
            mandatory = "必要" if skill.get("is_mandatory", 1) else "加分"
            skill_requirements.append(
                f"- {skill['name']}：要求熟练度{skill.get('required_level', 0):.0%}（{mandatory}）"
            )

        combined_text = (
            f"岗位：{title}\n\n岗位描述：\n{description}\n\n技能要求：\n"
            + "\n".join(skill_requirements)
        )

        chunks = self.text_splitter.split_text(combined_text)

        doc_id = f"pos_{position_id}"
        chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks)]
        metadatas = [
            {
                "position_id": position_id,
                "chunk_index": i,
                "type": "position_description",
            }
            for i in range(len(chunks)]
        ]

        collection = chroma_manager.get_collection("position_descriptions")
        try:
            collection.delete(where={"position_id": position_id})
        except Exception:
            pass

        collection.add(
            documents=chunks,
            ids=chunk_ids,
            metadatas=metadatas,
        )

        return doc_id

    def vectorize_knowledge_base(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        collection_name: str = "skill_knowledge",
    ) -> List[str]:
        """
        将知识库文档向量化
        Args:
            documents: 文档文本列表
            metadatas: 元数据列表
            collection_name: 铺合名称
        Returns:
            文档ID列表
        """
        all_chunks = []
        all_metadatas = []
        all_ids = []

        for i, doc in enumerate(documents):
            chunks = self.text_splitter.split_text(doc)
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_ids.append(f"kb_{uuid.uuid4().hex[:8]}_chunk_{j}")
                meta = metadatas[i] if metadatas and i < len(metadatas) else {}
                meta["chunk_index"] = j
                all_metadatas.append(meta)

        collection = chroma_manager.get_collection(collection_name)
        collection.add(
            documents=all_chunks,
            ids=all_ids,
            metadatas=all_metadatas,
        )

        return all_ids