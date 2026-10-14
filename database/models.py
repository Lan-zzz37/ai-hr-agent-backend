"""
SQLite 数据模型定义
使用 SQLAlchemy ORM 定义员工、岗位、技能等核心数据模型
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Text, DateTime,
    ForeignKey, Table, create_engine
)
from sqlalchemy.orm import relationship, declarative_base

Base = declarative_base()

# ==================== 关联表 ====================

# 员工-技能 多对多关联
employee_skill = Table(
    "employee_skill",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("employee_id", Integer, ForeignKey("employees.id"), nullable=False),
    Column("skill_id", Integer, ForeignKey("skills.id"), nullable=False),
    Column("proficiency_level", Float, default=0.0, comment="技能熟练度 0-1"),
    Column("years_of_experience", Float, default=0.0, comment="使用年限"),
    Column("updated_at", DateTime, default=datetime.utcnow, onupdate=datetime.utcnow),
)

# 岗位-技能 多对多关联
position_skill = Table(
    "position_skill",
    Base.metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("position_id", Integer, ForeignKey("positions.id"), nullable=False),
    Column("skill_id", Integer, ForeignKey("skills.id"), nullable=False),
    Column("required_level", Float, default=0.0, comment="要求熟练度 0-1"),
    Column("importance_weight", Float, default=1.0, comment="技能重要性权重"),
    Column("is_mandatory", Integer, default=1, comment="是否为必要技能 1=是 0=否"),
)


# ==================== 技能表 ====================
class Skill(Base):
    """技能字典表"""
    __tablename__ = "skills"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True, comment="技能名称")
    category = Column(String(50), comment="技能类别：技术/管理/软技能")
    description = Column(Text, comment="技能描述")
    parent_id = Column(Integer, ForeignKey("skills.id"), comment="父技能ID，用于技能树结构")
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    employees = relationship("Employee", secondary=employee_skill, back_populates="skills")
    positions = relationship("Position", secondary=position_skill, back_populates="skills")


# ==================== 员工表 ====================
class Employee(Base):
    """员工信息表"""
    __tablename__ = "employees"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(50), nullable=False, comment="姓名")
    employee_id = Column(String(20), unique=True, comment="工号")
    department = Column(String(50), comment="部门")
    current_position = Column(String(100), comment="当前岗位")
    years_of_service = Column(Float, default=0.0, comment="服务年限")
    education = Column(String(20), comment="学历")
    age = Column(Integer, comment="年龄")
    resume_text = Column(Text, comment="简历文本，用于向量化检索")
    profile_vector_id = Column(String(100), comment="Chroma中的向量ID")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    skills = relationship("Skill", secondary=employee_skill, back_populates="employees")
    match_results = relationship("MatchResult", back_populates="employee")


# ==================== 岗位表 ====================
class Position(Base):
    """岗位信息表"""
    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(100), nullable=False, comment="岗位名称")
    department = Column(String(50), comment="所属部门")
    level = Column(String(20), comment="岗位级别：初级/中级/高级")
    description = Column(Text, comment="岗位描述")
    responsibility = Column(Text, comment="岗位职责")
    is_intelligent_mfg = Column(Integer, default=0, comment="是否为智能制造相关岗位 1=是 0=否")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # 关系
    skills = relationship("Skill", secondary=position_skill, back_populates="positions")
    match_results = relationship("MatchResult", back_populates="position")


# ==================== 匹配结果表 ====================
class MatchResult(Base):
    """人岗匹配结果表"""
    __tablename__ = "match_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    employee_id = Column(Integer, ForeignKey("employees.id"), nullable=False)
    position_id = Column(Integer, ForeignKey("positions.id"), nullable=False)
    overall_score = Column(Float, comment="综合匹配分数 0-100")
    skill_match_score = Column(Float, comment="技能匹配分数")
    skill_gap_score = Column(Float, comment="技能差距分数")
    experience_score = Column(Float, comment="经验匹配分数")
    gap_details = Column(Text, comment="技能差距详情 JSON")
    reshaping_plan = Column(Text, comment="技能重塑建议 JSON")
    status = Column(String(20), default="pending", comment="状态：pending/completed")
    created_at = Column(DateTime, default=datetime.utcnow)

    # 关系
    employee = relationship("Employee", back_populates="match_results")
    position = relationship("Position", back_populates="match_results")


# ==================== 对话记录表 ====================
class ChatHistory(Base):
    """Agent 对话记录表"""
    __tablename__ = "chat_histories"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String(50), nullable=False, comment="会话ID")
    role = Column(String(20), nullable=False, comment="角色：user/assistant/system")
    content = Column(Text, nullable=False, comment="消息内容")
    metadata = Column(Text, comment="元数据 JSON")
    created_at = Column(DateTime, default=datetime.utcnow)