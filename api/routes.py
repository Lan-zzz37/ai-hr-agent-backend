"""
API 路由定义
包含所有 REST API 端点
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field
from typing import List, Optional
from backend.agent_core.agent import agent_session_manager
from backend.matching_engine.matcher import PositionMatcher
from backend.matching_engine.skill_gap import SkillGapAnalyzer
from backend.data_processor.cleaner import DataCleaner
from backend.data_processor.vectorizer import DataVectorizer
from backend.database.session import db_manager
from backend.database.models import Employee, Position, Skill, MatchResult

router = APIRouter()


# ==================== 请求/响应模型 ====================
class ChatRequest(BaseModel):
    session_id: str = Field(default="default", description="会话ID")
    message: str = Field(..., description="用户消息")


class ChatResponse(BaseModel):
    session_id: str
    reply: str


class MatchRequest(BaseModel):
    employee_id: int = Field(..., description="员工ID")
    position_ids: Optional[List[int]] = Field(default=None, description="目标岗位ID列表，为空则匹配所有")


class MatchResponse(BaseModel):
    employee_id: int
    matches: List[dict]


class SkillGapRequest(BaseModel):
    employee_id: int = Field(..., description="员工ID")
    position_id: int = Field(..., description="目标岗位ID")


class EmployeeCreate(BaseModel):
    name: str
    employee_id: str
    department: Optional[str] = ""
    current_position: Optional[str] = ""
    years_of_service: Optional[float] = 0
    education: Optional[str] = ""
    age: Optional[int] = 0
    resume_text: Optional[str] = ""


class PositionCreate(BaseModel):
    title: str
    department: Optional[str] = ""
    level: Optional[str] = "中级"
    description: Optional[str] = ""
    responsibility: Optional[str] = ""
    is_intelligent_mfg: Optional[int] = 0


class KnowledgeUploadRequest(BaseModel):
    documents: List[str] = Field(..., description="知识库文档列表")
    metadatas: Optional[List[dict]] = None


# ==================== Agent 对话接口 ====================
@router.post("/chat", response_model=ChatResponse, tags=["Agent对话"])
async def chat_with_agent(req: ChatRequest):
    """与AI顾问对话"""
    agent = agent_session_manager.get_agent(req.session_id)
    reply = await agent.chat(req.message)
    return ChatResponse(session_id=req.session_id, reply=reply)


@router.post("/chat/reset", tags=["Agent对话"])
async def reset_chat(session_id: str = Query(default="default")):
    """重置对话"""
    agent = agent_session_manager.get_agent(session_id)
    agent.reset_memory()
    return {"message": "对话已重置", "session_id": session_id}


# ==================== 人岗匹配接口 ====================
@router.post("/match", response_model=MatchResponse, tags=["人岗匹配"])
async def match_employee_to_positions(req: MatchRequest):
    """为员工匹配最佳岗位"""
    with db_manager.get_session() as session:
        # 查询员工
        employee = session.query(Employee).filter(Employee.id == req.employee_id).first()
        if not employee:
            raise HTTPException(status_code=404, detail="员工不存在")

        # 查询岗位
        if req.position_ids:
            positions = session.query(Position).filter(Position.id.in_(req.position_ids)).all()
        else:
            positions = session.query(Position).all()

        if not positions:
            raise HTTPException(status_code=404, detail="未找到目标岗位")

        # 构建匹配输入
        emp_data = {
            "id": str(employee.id),
            "skills": [{"name": s.name, "proficiency_level": 0.5} for s in employee.skills],
            "years_of_service": employee.years_of_service,
        }
        pos_list = []
        for pos in positions:
            pos_data = {
                "id": str(pos.id),
                "title": pos.title,
                "level": pos.level,
                "skills": [{"name": s.name, "required_level": 0.7, "importance_weight": 1.0, "is_mandatory": 1} for s in pos.skills],
            }
            pos_list.append(pos_data)

    matcher = PositionMatcher()
    results = matcher.match_employee_to_positions(emp_data, pos_list)

    return MatchResponse(
        employee_id=req.employee_id,
        matches=[r.to_dict() for r in results],
    )


# ==================== 技能差距分析接口 ====================
@router.post("/skill-gap", tags=["技能差距分析"])
async def analyze_skill_gap(req: SkillGapRequest):
    """分析员工与目标岗位的技能差距"""
    with db_manager.get_session() as session:
        employee = session.query(Employee).filter(Employee.id == req.employee_id).first()
        position = session.query(Position).filter(Position.id == req.position_id).first()

        if not employee:
            raise HTTPException(status_code=404, detail="员工不存在")
        if not position:
            raise HTTPException(status_code=404, detail="岗位不存在")

        emp_skills = [{"name": s.name, "proficiency_level": 0.5} for s in employee.skills]
        pos_skills = [{"name": s.name, "required_level": 0.7, "importance_weight": 1.0, "is_mandatory": 1} for s in position.skills]

    analyzer = SkillGapAnalyzer()
    result = analyzer.analyze(emp_skills, pos_skills, str(req.employee_id), str(req.position_id))
    return result.to_dict()


# ==================== 员工管理接口 ====================
@router.post("/employees", tags=["员工管理"])
async def create_employee(emp: EmployeeCreate):
    """创建员工"""
    with db_manager.get_session() as session:
        db_emp = Employee(
            name=emp.name,
            employee_id=emp.employee_id,
            department=emp.department,
            current_position=emp.current_position,
            years_of_service=emp.years_of_service,
            education=emp.education,
            age=emp.age,
            resume_text=emp.resume_text,
        )
        session.add(db_emp)
        session.flush()
        return {"id": db_emp.id, "message": "员工创建成功"}


@router.get("/employees", tags=["员工管理"])
async def listemployees(skip: int = 0, limit: int = 20):
    """获取员工列表"""
    with db_manager.get_session() as session:
        employees = session.query(Employee).offset(skip).limit(limit).all()
        return [
            {
                "id": e.id,
                "name": e.name,
                "employee_id": e.employee_id,
                "department": e.department,
                "current_position": e.current_position,
            }
            for e in employees
        ]


# ==================== 岗位管理接口 ====================
@router.post("/positions", tags=["岗位管理"])
async def create_position(pos: PositionCreate):
    """创建岗位"""
    with db_manager.get_session() as session:
        db_pos = Position(
            title=pos.title,
            department=pos.department,
            level=pos.level,
            description=pos.description,
            responsibility=pos.responsibility,
            is_intelligent_mfg=pos.is_intelligent_mfg,
        )
        session.add(db_pos)
        session.flush()
        return {"id": db_pos.id, "message": "岗位创建成功"}


@router.get("/positions", tags=["岗位管理"])
async def list_positions(skip: int = 0, limit: int = 20):
    """获取岗位列表"""
    with db_manager.get_session() as session:
        positions = session.query(Position).offset(skip).limit(limit).all()
        return [
            {
                "id": p.id,
                "title": p.title,
                "department": p.department,
                "level": p.level,
                "is_intelligent_mfg": p.is_intelligent_mfg,
            }
            for p in positions
        ]


# ==================== 知识库接口 ====================
@router.post("/knowledge/upload", tags=["知识库"])
async def upload_knowledge(req: KnowledgeUploadRequest):
    """上传知识库文档"""
    vectorizer = DataVectorizer()
    try:
        ids = vectorizer.vectorize_knowledge_base(
            documents=req.documents,
            metadatas=req.metadatas,
        )
        return {"message": "知识库上传成功", "document_count": len(ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"知识库上传失败：{str(e)}")


@router.get("/knowledge/stats", tags=["知识库"])
async def knowledge_stats():
    """获取知识库统计信息"""
    from backend.database.chroma_client import chroma_manager
    stats = chroma_manager.get_collection_stats()
    return stats


# ==================== 健康检查 ====================
@router.get("/health", tags=["系统"])
async def health_check():
    """系统健康检查"""
    return {
        "status": "healthy",
        "service": "AI人岗匹配与技能重塑Agent",
        "version": "1.0.0",
    }