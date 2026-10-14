"""
数据库会话管理
提供 SQLite 连接和会话的统一管理
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from backend.config import SQLITE_DB_PATH
from backend.database.models import Base


class DatabaseManager:
    """数据库管理器，单例模式"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_engine()
        return cls._instance

    def _init_engine(self):
        """初始化数据库引擎"""
        self.engine = create_engine(
            f"sqlite:///{SQLITE_DB_PATH}",
            echo=False,
            connect_args={"check_same_thread": False},
        )
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

    def create_tables(self):
        """创建所有数据表"""
        Base.metadata.create_all(bind=self.engine)

    def drop_tables(self):
        """删除所有数据表（慎用）"""
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self) -> Session:
        """获取数据库会话的上下文管理器"""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def get_db_session(self) -> Session:
        """获取数据库会话（需手动关闭）"""
        return self.SessionLocal()


# 全局数据库管理器实例
db_manager = DatabaseManager()


def init_database():
    """初始化数据库：创建表结构"""
    db_manager.create_tables()
    return db_manager