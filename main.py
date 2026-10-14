"""
FastAPI 应用入口
面向智能制造转型的AI人岗匹配与技能重塑Agent
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from backend.config import API_TITLE, API_VERSION, API_DESCRIPTION, CORS_ORIGINS
from backend.database.session import init_database
from backend.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化数据库
    print("正在初始化数据库...")
    init_database()
    print("数据库初始化完成")

    # 初始化 ChromaDB 集合
    print("正在初始化向量数据库...")
    from backend.database.chroma_client import chroma_manager
    chroma_manager.get_collection("skill_knowledge")
    chroma_manager.get_collection("employee_profiles")
    chroma_manager.get_collection("position_descriptions")
    print("向量数据库初始化完成")

    print(f"🚀 {API_TITLE} v{API_VERSION} 启动成功！")
    yield

    # 关闭时清理资源
    print("正在关闭服务...")


# 创建 FastAPI 应用
app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    description=API_DESCRIPTION,
    lifespan=lifespan,
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "redoc": "/redoc",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )