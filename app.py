"""
AI人岗匹配与技能重塑Agent — FastAPI 应用入口
================================================
独立启动：cd backend && python app.py
或：uvicorn app:app --reload --port 8000
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
# 确保 backend 目录在 sys.path 中，以便 import config / matching_engine
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except NameError:
    sys.path.insert(0, os.getcwd())

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import uvicorn

import config
import matching_engine
from matching_agent import SmartMfgAgent, agent_manager


# ==================== FastAPI 应用 ====================

app = FastAPI(
    title=config.API_TITLE,
    version=config.API_VERSION,
    description=config.API_DESCRIPTION,
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== 请求/响应模型 ====================

class ChatRequest(BaseModel):
    message: str = Field(..., description="用户消息")
    role: str = Field(default="employee", description="角色：employee 或 hr")
    session_id: str = Field(default="web-session", description="会话ID")

class ChatResponse(BaseModel):
    reply: str
    role: str
    session_id: str

class ChatResetRequest(BaseModel):
    role: str = Field(default="employee", description="角色")
    session_id: str = Field(default="web-session", description="会话ID")

class MatchRequest(BaseModel):
    employee_skills: str = Field(..., description="员工现有技能描述")
    target_job_skills: str = Field(..., description="目标岗位技能要求描述")
    target_job_name: str = Field(default="", description="目标岗位名称")

class MatchResponse(BaseModel):
    score: float
    gap: dict
    training: list

class SkillGapRequest(BaseModel):
    employee_skills: str = Field(..., description="员工现有技能描述")
    target_job_skills: str = Field(..., description="目标岗位技能要求描述")

class TrainingPathRequest(BaseModel):
    missing_skills: List[str] = Field(..., description="缺失技能列表")
    target_job: str = Field(..., description="目标岗位名称")


# ==================== API 端点 ====================

@app.get("/api/dashboard/stats")
async def dashboard_stats():
    """Dashboard 统计数据"""
    return {
        "employee_count": 3,
        "position_count": 3,
        "knowledge_count": 25,
        "system_status": "正常",
        "llm_model": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
    }


@app.get("/api/employees")
async def list_employees(limit: int = 20, offset: int = 0):
    """员工列表（模拟数据）"""
    employees = [
        {"id": "E001", "name": "张建国", "role": "质检员", "skills": "质量检验,缺陷识别,SPC统计过程控制,六西格玛", "experience": "10年"},
        {"id": "E002", "name": "李明", "role": "钳工", "skills": "机械加工,精密测量,图纸识读,数控操作", "experience": "5年"},
        {"id": "E003", "name": "王芳", "role": "组装工", "skills": "机械装配,工艺流程,设备操作,质量控制", "experience": "3年"},
    ]
    return employees[offset:offset+limit]


@app.get("/api/positions")
async def list_positions(limit: int = 20, offset: int = 0):
    """岗位列表（模拟数据）"""
    positions = [
        {"id": "P001", "name": "AI视觉检测员", "skills": "机器视觉,图像识别,深度学习,Python编程,缺陷检测算法"},
        {"id": "P002", "name": "机器人运维员", "skills": "工业机器人操作,PLC编程,故障诊断,预防性维护,机器人编程"},
        {"id": "P003", "name": "自动化设备操作员", "skills": "自动化装配,传感器技术,MES系统,SCADA系统,数据分析"},
    ]
    return positions[offset:offset+limit]


@app.get("/api/knowledge/stats")
async def knowledge_stats():
    """知识库统计"""
    return {"total": 12, "indexed": 10, "pending": 2}


@app.get("/api/health")
async def health_check():
    """健康检查"""
    return {
        "status": "ok",
        "title": config.API_TITLE,
        "version": config.API_VERSION,
        "llm_model": config.LLM_MODEL_NAME,
        "embedding_model": config.EMBEDDING_MODEL_NAME,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    与 AI Agent 对话
    - role="employee" → 员工端（匹配分析 + 心理疏导）
    - role="hr"       → HR端（汇总报告 + 人力优化）
    """
    try:
        agent = agent_manager.get_agent(session_id=req.session_id, role=req.role)
        reply = agent.chat(req.message)
        return ChatResponse(reply=reply, role=req.role, session_id=req.session_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话出错：{str(e)}")


@app.post("/api/chat/reset")
async def reset_chat(req: ChatResetRequest):
    """重置对话历史"""
    agent_manager.reset_session(session_id=req.session_id, role=req.role)
    return {"status": "ok", "message": "对话已重置", "role": req.role, "session_id": req.session_id}


@app.post("/api/match/semantic", response_model=MatchResponse)
async def semantic_match(req: MatchRequest):
    """
    语义匹配：计算员工技能与岗位需求的匹配度
    调用 matching_engine.calculate_match_score + analyze_skill_gap + generate_training_path
    """
    try:
        # 1. 匹配度得分
        score = matching_engine.calculate_match_score(req.employee_skills, req.target_job_skills)

        # 2. 技能差距分析
        gap = matching_engine.analyze_skill_gap(req.employee_skills, req.target_job_skills)

        # 3. 培训路径（如果有缺失技能）
        training = []
        if gap.get("missing_skills"):
            target = req.target_job_name or "目标岗位"
            training = matching_engine.generate_training_path(gap["missing_skills"], target)

        return MatchResponse(score=score, gap=gap, training=training)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"匹配分析出错：{str(e)}")


@app.post("/api/match/skill-gap")
async def skill_gap_analysis(req: SkillGapRequest):
    """技能差距分析"""
    try:
        result = matching_engine.analyze_skill_gap(req.employee_skills, req.target_job_skills)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"技能差距分析出错：{str(e)}")


@app.post("/api/match/training-path")
async def training_path(req: TrainingPathRequest):
    """转岗路径规划"""
    try:
        result = matching_engine.generate_training_path(req.missing_skills, req.target_job)
        return {"courses": result, "total_courses": len(result), "target_job": req.target_job}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"培训路径生成出错：{str(e)}")


@app.get("/api/match/courses")
async def get_course_library():
    """获取培训课程库"""
    return {"courses": matching_engine.TRAINING_COURSE_LIBRARY, "total": len(matching_engine.TRAINING_COURSE_LIBRARY)}


# ==================== 启动入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print(f"  {config.API_TITLE} v{config.API_VERSION}")
    print(f"  LLM: {config.LLM_MODEL_NAME}")
    print(f"  Embedding: {config.EMBEDDING_MODEL_NAME}")
    print(f"  API Docs: http://localhost:8000/docs")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)