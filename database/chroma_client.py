"""
Chroma 向量数据库客户端
用于知识库的存储和语义检索
"""
import chromadb
from typing import List, Dict, Optional
from backend.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME


class ChromaManager:
    """ChromaDB 向量数据库管理器"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_client()
        return cls._instance

    def _init_client(self):
        """初始化 ChromaDB 客户端"""
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self._collections: Dict[str, chromadb.Collection] = {}

    def get_collection(self, collection_name: Optional[str] = None) -> chromadb.Collection:
        """获取或创建集合"""
        name = collection_name or CHROMA_COLLECTION_NAME
        if name not in self._collections:
            self._collections[name] = self.client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[name]

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        添加文档到向量库
        Args:
            documents: 文档文本列表
            metadatas: 元数据列表
            ids: 文档ID列表
            collection_name: 集合名称
        """
        collection = self.get_collection(collection_name)
        if ids is None:
            ids = [f"doc_{collection.count() + i}" for i in range(len(documents))]
        collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids,
        )

    def query(
        self,
        query_texts: List[str],
        n_results: int = 5,
        where: Optional[Dict] = None,
        collection_name: Optional[str] = None,
    ) -> Dict:
        """
        语义检索
        Args:
            query_texts: 查询文本列表
            n_results: 返回结果数量
            where: 元数据过滤条件
            collection_name: 集合名称
        Returns:
            检索结果字典
        """
        collection = self.get_collection(collection_name)
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
        )

    def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict] = None,
        collection_name: Optional[str] = None,
    ) -> None:
        """删除文档"""
        collection = self.get_collection(collection_name)
        collection.delete(ids=ids, where=where)

    def get_collection_stats(self, collection_name: Optional[str] = None) -> Dict:
        """获取集合统计信息"""
        collection = self.get_collection(collection_name)
        return {
            "name": collection.name,
            "count": collection.count(),
        }


# 全局 ChromaDB 管理器实例
chroma_manager = ChromaManager()