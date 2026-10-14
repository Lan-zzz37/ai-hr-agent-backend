"""
人岗匹配与技能重塑 核心算法模块
================================
三大核心功能：
1. calculate_match_score  — 语义匹配（Embedding + 余弦相似度）
2. analyze_skill_gap      — 技能差距分析（LLM 驱动）
3. generate_training_path — 转岗路径规划（LLM + 课程库）

用法：cd backend && python matching_engine.py
"""
from dotenv import load_dotenv
load_dotenv()

import math
import json
from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
import config

# ==================== 1. 语义匹配算法 ====================

def get_embedding(text: str) -> list:
    """调用阿里云百炼 Embedding API，将文本转为向量"""
    client = OpenAI(
        api_key=config.EMBEDDING_API_KEY,
        base_url=config.EMBEDDING_API_BASE,
    )
    response = client.embeddings.create(
        model=config.EMBEDDING_MODEL_NAME,
        input=text,
    )
    return response.data[0].embedding


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """计算两个向量的余弦相似度"""
    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def calculate_match_score(employee_skills: str, target_job_skills: str) -> float:
    """
    语义匹配算法：计算员工技能与岗位需求的匹配度
    Args:
        employee_skills:   员工现有技能描述（纯字符串）
        target_job_skills: 目标岗位技能要求描述（纯字符串）
    Returns:
        匹配度得分 0-100
    """
    # 分别向量化
    emp_vector = get_embedding(employee_skills)
    job_vector = get_embedding(target_job_skills)

    # 余弦相似度 → 0~100
    similarity = cosine_similarity(emp_vector, job_vector)
    score = max(0.0, min(100.0, similarity * 100))
    return round(score, 2)


# ==================== 2. 技能差距分析（SKGap）====================

def analyze_skill_gap(employee_skills: str, target_job_skills: str) -> dict:
    """
    技能差距分析：调用 LLM 明确列出已具备/缺失的技能及匹配结论
    Args:
        employee_skills:   员工现有技能描述
        target_job_skills: 目标岗位技能要求描述
    Returns:
        {
            "matched_skills": [...],    # 已具备的技能
            "missing_skills": [...],    # 缺失的核心技能
            "gap_severity": str,        # 整体差距等级 low/medium/high/critical
            "conclusion": str           # 匹配度结论
        }
    """
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0.3,
        max_tokens=1024,
    )

    system_prompt = """你是一位智能制造领域的技能评估专家。请对比员工的现有技能和目标岗位的技能要求，进行严格的技能差距分析。

请严格按照以下 JSON 格式输出，不要输出任何其他内容：
{
    "matched_skills": ["已具备的技能1", "已具备的技能2"],
    "missing_skills": ["缺失的核心技能1", "缺失的核心技能2"],
    "gap_severity": "low/medium/high/critical",
    "conclusion": "匹配度结论描述"
}

差距等级判定标准：
- low: 缺失1项非核心技能
- medium: 缺失1-2项核心技能
- high: 缺失3项及以上核心技能
- critical: 技能方向差异极大，需系统性重塑"""

    user_prompt = f"""请分析以下技能差距：

【员工现有技能】
{employee_skills}

【目标岗位技能要求】
{target_job_skills}

请输出 JSON 格式的差距分析结果。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    # 解析 LLM 返回的 JSON
    content = response.content.strip()
    # 去除可能的 markdown 代码块标记
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        # 如果解析失败，返回原始文本
        result = {
            "matched_skills": [],
            "missing_skills": [],
            "gap_severity": "unknown",
            "conclusion": content,
        }

    return result


# ==================== 3. 转岗路径规划 ====================

# 公开培训课程库（模拟数据）
TRAINING_COURSE_LIBRARY = [
    {"course_id": "TC001", "name": "工业视觉检测基础", "category": "AI视觉", "hours": 40,
     "skills": ["机器视觉", "图像识别", "缺陷检测"], "level": "初级",
     "description": "涵盖工业相机选型、光源设计、图像采集与预处理基础"},
    {"course_id": "TC002", "name": "深度学习与图像分类", "category": "AI视觉", "hours": 60,
     "skills": ["深度学习", "CNN", "图像分类"], "level": "中级",
     "description": "从神经网络基础到CNN图像分类实战，含PyTorch框架"},
    {"course_id": "TC003", "name": "PLC编程与自动化控制", "category": "自动化", "hours": 48,
     "skills": ["PLC编程", "自动化控制", "西门子S7"], "level": "初级",
     "description": "西门子S7-1200/1500 PLC编程基础与HMI配置"},
    {"course_id": "TC004", "name": "工业机器人运维实操", "category": "机器人", "hours": 56,
     "skills": ["机器人运维", "故障诊断", "安全规范"], "level": "中级",
     "description": "六轴机器人日常维护、故障排查与安全操作规范"},
    {"course_id": "TC005", "name": "Python数据分析与可视化", "category": "数据科学", "hours": 36,
     "skills": ["Python", "数据分析", "数据可视化"], "level": "初级",
     "description": "Python基础、Pandas数据处理、Matplotlib可视化"},
    {"course_id": "TC006", "name": "数字孪生技术与应用", "category": "数字化", "hours": 52,
     "skills": ["数字孪生", "仿真建模", "虚实互联"], "level": "高级",
     "description": "数字孪生体构建、实时数据驱动仿真与预测性维护"},
    {"course_id": "TC007", "name": "质量管理体系与SPC", "category": "质量管理", "hours": 32,
     "skills": ["质量管理体系", "SPC", "六西格玛"], "level": "中级",
     "description": "统计过程控制(SPC)工具应用与六西格玛绿带基础"},
    {"course_id": "TC008", "name": "AI缺陷检测实战", "category": "AI视觉", "hours": 64,
     "skills": ["缺陷检测", "目标检测", "模型部署"], "level": "高级",
     "description": "基于YOLOv8的表面缺陷检测模型训练、优化与边缘部署"},
    {"course_id": "TC009", "name": "MES系统操作与数据采集", "category": "数字化", "hours": 28,
     "skills": ["MES系统", "数据采集", "SCADA"], "level": "初级",
     "description": "制造执行系统操作、设备数据采集与SCADA集成"},
    {"course_id": "TC010", "name": "智能制造转型管理", "category": "管理", "hours": 24,
     "skills": ["转型规划", "变革管理", "精益生产"], "level": "高级",
     "description": "企业智能制造转型路径规划与变革管理方法论"},
]


def generate_training_path(missing_skills: list, target_job: str) -> list:
    """
    转岗路径规划：根据缺失技能推荐培训课程
    Args:
        missing_skills: 缺失的技能列表
        target_job:     目标岗位名称
    Returns:
        推荐课程列表，每项含 course_id/name/hours/reason/matched_skills
    """
    llm = ChatOpenAI(
        model=config.LLM_MODEL_NAME,
        api_key=config.LLM_API_KEY,
        base_url=config.LLM_API_BASE,
        temperature=0.3,
        max_tokens=1024,
    )

    # 构建课程库摘要供 LLM 参考
    course_catalog = "\n".join(
        f"  [{c['course_id']}] {c['name']} | {c['hours']}学时 | 覆盖技能: {', '.join(c['skills'])} | {c['description']}"
        for c in TRAINING_COURSE_LIBRARY
    )

    system_prompt = """你是一位智能制造培训规划专家。根据员工缺失的技能，从课程库中推荐最合适的3门核心课程。

请严格按照以下 JSON 数组格式输出，不要输出任何其他内容：
[
    {
        "course_id": "课程编号",
        "reason": "推荐理由"
    },
    ...
]

要求：
1. 只推荐课程库中存在的课程
2. 优先推荐能覆盖多项缺失技能的课程
3. 按重要度排序，最重要的排前面
4. 推荐理由需说明该课程如何补齐缺失技能"""

    user_prompt = f"""缺失技能：{', '.join(missing_skills)}
目标岗位：{target_job}

可选课程库：
{course_catalog}

请推荐3门最合适的课程。"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    content = response.content.strip()
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()
    if content.endswith("```"):
        content = content[:-3].strip()

    try:
        llm_recommendations = json.loads(content)
    except json.JSONDecodeError:
        llm_recommendations = []

    # 将 LLM 推荐与课程库详情合并
    course_map = {c["course_id"]: c for c in TRAINING_COURSE_LIBRARY}
    result = []
    for rec in llm_recommendations[:3]:
        cid = rec.get("course_id", "")
        course = course_map.get(cid)
        if course:
            result.append({
                "course_id": cid,
                "name": course["name"],
                "hours": course["hours"],
                "level": course["level"],
                "matched_skills": course["skills"],
                "reason": rec.get("reason", ""),
            })

    # 如果 LLM 推荐不足3门，从课程库按技能匹配度补充
    if len(result) < 3:
        recommended_ids = {r["course_id"] for r in result}
        for course in TRAINING_COURSE_LIBRARY:
            if course["course_id"] in recommended_ids:
                continue
            overlap = len(set(course["skills"]) & set(missing_skills))
            if overlap > 0:
                result.append({
                    "course_id": course["course_id"],
                    "name": course["name"],
                    "hours": course["hours"],
                    "level": course["level"],
                    "matched_skills": course["skills"],
                    "reason": f"覆盖{overlap}项缺失技能：{', '.join(set(course['skills']) & set(missing_skills))}",
                })
                recommended_ids.add(course["course_id"])
            if len(result) >= 3:
                break

    return result


# ==================== 测试入口 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("  人岗匹配与技能重塑 — 核心算法测试")
    print("=" * 60)

    # 测试场景：质检员 → AI视觉检测员
    employee_skills = "10年质检经验，精通质量检验标准与流程，熟悉SPC统计过程控制，掌握测量仪器操作，了解ISO9001质量管理体系"
    target_job_skills = "机器视觉系统开发，深度学习图像分类，工业缺陷检测算法，Python编程，相机与光源选型，模型部署与优化"

    # ---- 1. 语义匹配 ----
    print("\n【1】语义匹配算法")
    print("-" * 60)
    print(f"员工技能: {employee_skills}")
    print(f"岗位要求: {target_job_skills}")
    score = calculate_match_score(employee_skills, target_job_skills)
    print(f"\n✅ 匹配度得分: {score} / 100")

    # ---- 2. 技能差距分析 ----
    print("\n【2】技能差距分析 (SKGap)")
    print("-" * 60)
    gap_result = analyze_skill_gap(employee_skills, target_job_skills)
    print(f"已具备技能: {gap_result.get('matched_skills', [])}")
    print(f"缺失技能:   {gap_result.get('missing_skills', [])}")
    print(f"差距等级:   {gap_result.get('gap_severity', 'N/A')}")
    print(f"结论:       {gap_result.get('conclusion', 'N/A')}")

    # ---- 3. 转岗路径规划 ----
    print("\n【3】转岗路径规划")
    print("-" * 60)
    missing = gap_result.get("missing_skills", [])
    if missing:
        training_path = generate_training_path(missing, "AI视觉检测员")
        for i, course in enumerate(training_path, 1):
            print(f"\n  推荐{i}: {course['name']} [{course.get('level', '')}]")
            print(f"    课程编号: {course['course_id']}")
            print(f"    预计学时: {course['hours']}小时")
            print(f"    覆盖技能: {', '.join(course.get('matched_skills', []))}")
            print(f"    推荐理由: {course.get('reason', '')}")
    else:
        print("  无缺失技能，无需培训推荐。")

    print("\n" + "=" * 60)
    print("  测试完成")
    print("=" * 60)