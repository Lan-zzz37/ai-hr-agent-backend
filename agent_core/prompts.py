"""
提示词模板管理
定义 Agent 各功能场景的提示词模板
"""

# ==================== 系统角色提示词 ====================
SYSTEM_ROLE_PROMPT = """你是"智能制造转型AI顾问"，一个专业的智能制造领域人岗匹配与技能重塑助手。

你的核心能力：
1. 分析员工技能与目标岗位的匹配程度
2. 识别技能差距并评估严重程度
3. 制定个性化的技能重塑路径和学习计划
4. 提供智能制造行业转型建议

你的知识范围：
- 智能制造核心技术：工业物联网(IIoT)、数字孪生、MES/ERP系统、工业机器人、机器视觉、CNC数控
- 智能制造管理：精益生产、质量管理体系、供应链数字化
- 技能评估方法论：能力矩阵、技能画像、差距分析模型

回答要求：
- 专业且易懂，避免过于学术化
- 给出具体可执行的建议
- 结合行业最佳实践
- 关注员工职业发展和企业转型需求的双向匹配
"""

# ==================== 人岗匹配分析提示词 ====================
MATCHING_ANALYSIS_PROMPT = """请分析以下员工与目标岗位的匹配情况：

## 员工信息
- 姓名：{employee_name}
- 当前岗位：{current_position}
- 工作年限：{years_of_service}年
- 现有技能：{employee_skills}

## 目标岗位
- 岗位名称：{position_title}
- 岗位级别：{position_level}
- 技能要求：{position_skills}
- 岗位职责：{responsibility}

## 技能差距分析数据
{gap_analysis_data}

请从以下维度给出分析：
1. **匹配度评估**：综合评估员工的岗位适配程度
2. **优势分析**：员工已有的、能迁移到新岗位的技能优势
3. **关键差距**：需要重点补齐的技能短板
4. **转型建议**：从当前岗位到目标岗位的转型路径建议

请以结构化的JSON格式返回分析结果。"""

# ==================== 技能重塑计划提示词 ====================
SKILL_RESHAPING_PROMPT = """基于以下技能差距分析，请制定详细的技能重塑计划：

## 员工信息
- 姓名：{employee_name}
- 当前技能水平：{current_skills}
- 目标岗位：{target_position}
- 关键差距技能：{gap_skills}

## 行业背景
该企业正在进行智能制造转型，需要将传统制造岗位人员转型为数字化、智能化岗位。

请制定包含以下内容的技能重塑计划：

1. **学习路径**：按优先级排列的学习模块
   - 模块名称
   - 学习目标
   - 预计学习时长（小时）
   - 推荐学习资源类型（在线课程/实操训练/项目实践/导师辅导）

2. **阶段规划**：
   - 第一阶段（1-3个月）：基础技能补齐
   - 第二阶段（3-6个月）：核心技能提升
   - 第三阶段（6-12个月）：综合能力强化

3. **评估节点**：每个阶段结束时的技能评估标准

4. **风险提示**：学习过程中可能遇到的困难和应对建议

请以结构化的JSON格式返回计划。"""

# ==================== 行业知识问答提示词 ====================
KNOWLEDGE_QA_PROMPT = """基于以下知识库内容，回答用户关于智能制造转型的问题：

## 相关知识库内容
{retrieved_context}

## 用户问题
{user_question}

请结合知识库内容和你的专业知识，给出准确、实用的回答。如果知识库中没有相关信息，请基于你的专业知识回答，并注明"以下建议基于行业通用实践"。"""

# ==================== 技能推荐提示词 ====================
SKILL_RECOMMENDATION_PROMPT = """基于以下员工的技能画像，推荐适合的智能制造转型方向：

## 员工技能画像
- 当前岗位：{current_position}
- 核心技能：{core_skills}
- 工作经验：{experience_summary}
- 兴趣方向：{interest_direction}

## 智能制造热门方向
1. 工业数据分析与AI应用
2. 数字孪生与仿真
3. 工业物联网平台运维
4. 智能产线规划与优化
5. 机器视觉质检
6. 智能仓储与物流

请为该员工推荐2-3个最适合的转型方向，并说明：
- 推荐理由（与现有技能的关联度）
- 需要新增的核心技能
- 预计转型周期
- 职业发展前景"""

# ==================== 提示词模板注册 ====================
PROMPT_TEMPLATES = {
    "system_role": SYSTEM_ROLE_PROMPT,
    "matching_analysis": MATCHING_ANALYSIS_PROMPT,
    "skill_reshaping": SKILL_RESHAPING_PROMPT,
    "knowledge_qa": KNOWLEDGE_QA_PROMPT,
    "skill_recommendation": SKILL_RECOMMENDATION_PROMPT,
}


def get_prompt(template_name: str, **kwargs) -> str:
    """
    获取并填充提示词模板
    Args:
        template_name: 模板名称
        **kwargs: 模板变量
    Returns:
        填充后的提示词文本
    """
    template = PROMPT_TEMPLATES.get(template_name, "")
    if not template:
        raise ValueError(f"未找到提示词模板：{template_name}")
    return template.format(**kwargs)


def list_templates() -> list:
    """列出所有可用的提示词模板名称"""
    return list(PROMPT_TEMPLATES.keys())