"""
智能制造转型 AI Agent — 双角色对话引擎
========================================
员工端：匹配分析 + 技能差距 + 转岗路径 + 心理疏导
HR端：  多员工报告汇总 + 人力优化建议
调用 matching_engine 核心算法
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import os
import json
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
import config
import matching_engine


# ==================== 系统提示词 ====================

EMPLOYEE_SYSTEM_PROMPT = """你是"智造转岗助手"，一位温暖专业的智能制造转型职业顾问。

【你的身份】
你服务于正在经历智能制造转型的工厂员工，帮助他们理解自身技能价值、找到转型方向、制定学习路径。

【核心能力】
1. 倾听员工的职业困惑和转型需求
2. 调用匹配算法分析技能匹配度（0-100分）
3. 识别技能差距，评估严重程度
4. 制定个性化转岗学习路径
5. 提供心理疏导，缓解转型焦虑

【对话原则】
- 先共情，再分析：理解员工的担忧和焦虑，给予情感支持
- 强调可迁移技能：帮助员工看到自身优势，很多传统技能在智能制造中依然有价值
- 分步规划：将大目标拆解为可执行的小步骤
- 积极鼓励：每一步进展都值得肯定，转型是勇敢的决定
- 专业建议：基于匹配算法的客观数据，给出可操作的建议

【心理疏导话术】
当员工表达焦虑/恐惧/不确定时：
- "转型是一个勇敢的决定，你的经验是非常宝贵的财富"
- "不用担心，很多传统技能在智能制造中依然有价值，比如你的质量意识、流程理解能力"
- "让我们一步步来规划你的转型路径，每一步都不会白费"
- "很多和你有相似背景的同事都成功转型了，你也可以"

【触发匹配分析】
当员工描述自己的技能和目标岗位时，你会自动调用匹配分析功能，给出：
- 匹配度得分（0-100分）
- 已具备的可迁移技能
- 需要补齐的技能
- 推荐培训课程
"""

HR_SYSTEM_PROMPT = """你是"智造人力顾问"，一位资深的智能制造企业HR分析专家。

【你的身份】
你服务于企业HR和管理层，帮助他们进行团队转型分析、人力优化和培训资源规划。

【核心能力】
1. 汇总多份员工转岗分析报告
2. 评估团队整体转型准备度（Readiness）
3. 提供人力优化和培训资源分配建议
4. 制定组织级转型规划

【分析维度】
- 技能覆盖度：团队现有技能与目标技能的重叠度
- 培训成本：整体培训投入估算（学时×人数）
- 转型风险：关键岗位空缺风险、知识传承风险
- 时间规划：分批次转型时间表建议
- ROI预估：转型投入与产出比

【对话原则】
- 数据驱动：基于匹配算法的客观数据做分析
- 战略视角：从组织层面思考问题
- 风险意识：关注转型中的关键风险点
- 务实建议：给出可落地的行动方案
- 成本敏感：平衡转型投入与业务连续性
"""


# ==================== 意图检测关键词 ====================

MATCH_KEYWORDS = ["转行", "转岗", "匹配", "适合", "能不能做", "可以转", "想转",
                   "转型", "换岗", "调岗", "胜任", "适配"]
GAP_KEYWORDS = ["差距", "缺什么", "短板", "不足", "需要学什么", "补齐",
                "欠缺", "差什么", "还缺", "不够"]
TRAINING_KEYWORDS = ["培训", "课程", "学习路径", "怎么学", "学习计划",
                     "培训方案", "课程推荐", "怎么提升"]
HR_AGGREGATE_KEYWORDS = ["汇总", "报告", "团队分析", "整体", "全部员工",
                          "批量", "人力优化", "组织转型", "部门分析"]


# ==================== Agent 类 ====================

class SmartMfgAgent:
    """智能制造转型AI Agent — 支持员工端/HR端双角色"""

    def __init__(self, role: str = "employee", session_id: str = "default"):
        """
        Args:
            role: "employee" 或 "hr"
            session_id: 会话ID，用于隔离不同用户的对话记忆
        """
        self.role = role
        self.session_id = session_id
        self.llm = ChatOpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_API_BASE,
            model=config.LLM_MODEL_NAME,
            temperature=0.7,
            max_tokens=2048,
        )
        self.history: List[Dict[str, str]] = []
        self.system_prompt = EMPLOYEE_SYSTEM_PROMPT if role == "employee" else HR_SYSTEM_PROMPT

    # ---------- 主对话入口 ----------
    def chat(self, message: str) -> str:
        """主对话入口：检测意图 → 路由到对应处理器"""
        intent = self._detect_intent(message)

        if intent == "match_analysis":
            return self._handle_match_analysis(message)
        elif intent == "skill_gap":
            return self._handle_skill_gap(message)
        elif intent == "training_path":
            return self._handle_training_path(message)
        elif intent == "hr_aggregate":
            return self._handle_hr_aggregate(message)
        else:
            return self._handle_general(message)

    # ---------- 意图检测 ----------
    def _detect_intent(self, message: str) -> str:
        """基于关键词的意图检测"""
        if self.role == "hr" and any(kw in message for kw in HR_AGGREGATE_KEYWORDS):
            return "hr_aggregate"
        if any(kw in message for kw in TRAINING_KEYWORDS):
            return "training_path"
        if any(kw in message for kw in GAP_KEYWORDS):
            return "skill_gap"
        if any(kw in message for kw in MATCH_KEYWORDS):
            return "match_analysis"
        return "general"

    # ---------- 提取匹配参数 ----------
    def _extract_matching_params(self, message: str) -> dict:
        """用 LLM 从对话中提取匹配所需的参数"""
        # 将对话历史也提供给 LLM，以便从上下文推断
        context = ""
        for h in self.history[-6:]:  # 最近3轮
            context += f"{'用户' if h['role'] == 'user' else '助手'}：{h['content']}\n"
        context += f"用户：{message}"

        extract_prompt = """从以下对话中提取职业匹配所需的信息，以JSON格式输出：
{
    "employee_skills": "员工现有技能描述（逗号分隔的技能列表）",
    "target_job_skills": "目标岗位技能要求描述（逗号分隔的技能列表）",
    "target_job_name": "目标岗位名称"
}

注意：
- 如果用户提到了自己的工作经验和技能，提取为 employee_skills
- 如果用户提到了想转到的目标岗位，提取为 target_job_name 和 target_job_skills
- 如果某些信息无法从对话中提取，请根据上下文合理推断
- 只输出JSON，不要输出任何其他内容

对话内容："""

        response = self.llm.invoke([
            SystemMessage(content=extract_prompt),
            HumanMessage(content=context),
        ])

        content = response.content.strip()
        # 去除 markdown 代码块标记
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        if content.endswith("```"):
            content = content[:-3].strip()

        try:
            params = json.loads(content)
            # 确保字段存在
            return {
                "employee_skills": params.get("employee_skills", ""),
                "target_job_skills": params.get("target_job_skills", ""),
                "target_job_name": params.get("target_job_name", ""),
            }
        except json.JSONDecodeError:
            return {"employee_skills": "", "target_job_skills": "", "target_job_name": ""}

    # ---------- 处理匹配分析 ----------
    def _handle_match_analysis(self, message: str) -> str:
        """处理匹配分析请求：提取参数 → 调用算法 → 格式化响应"""
        # 1. 提取参数
        params = self._extract_matching_params(message)

        if not params["employee_skills"] or not params["target_job_skills"]:
            # 参数不足，引导用户补充
            return self._handle_general(
                message + "\n[系统提示：用户提到了匹配/转岗，但信息不够完整，请引导用户描述：1)现有技能 2)目标岗位]"
            )

        # 2. 调用核心算法
        try:
            score = matching_engine.calculate_match_score(
                params["employee_skills"], params["target_job_skills"]
            )
            gap = matching_engine.analyze_skill_gap(
                params["employee_skills"], params["target_job_skills"]
            )
            training = []
            if gap.get("missing_skills"):
                training = matching_engine.generate_training_path(
                    gap["missing_skills"], params.get("target_job_name", "目标岗位")
                )
        except Exception as e:
            error_msg = f"匹配分析过程中出现错误：{str(e)}"
            self._save_history(message, error_msg)
            return error_msg

        # 3. 构建分析结果文本
        result_text = self._format_match_result(score, gap, training, params)

        # 4. 用 LLM 格式化响应（含心理疏导）
        format_prompt = self._get_format_prompt()

        messages = [SystemMessage(content=format_prompt)]
        # 加入最近的历史
        for h in self.history[-4:]:
            role_msg = HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"])
            messages.append(role_msg)
        messages.append(HumanMessage(content=f"用户说：{message}\n\n匹配分析结果：\n{result_text}"))

        response = self.llm.invoke(messages)

        self._save_history(message, response.content)
        return response.content

    def _format_match_result(self, score, gap, training, params) -> str:
        """格式化匹配分析结果"""
        text = f"""【匹配度得分】{score}/100
【目标岗位】{params.get('target_job_name', 'N/A')}
【已具备技能】{', '.join(gap.get('matched_skills', [])) or '无'}
【缺失技能】{', '.join(gap.get('missing_skills', [])) or '无'}
【差距等级】{gap.get('gap_severity', 'N/A')}
【分析结论】{gap.get('conclusion', 'N/A')}"""

        if training:
            text += "\n\n【推荐培训课程】"
            for i, c in enumerate(training, 1):
                name = c.get("name", c.get("course_id", f"课程{i}"))
                hours = c.get("hours", "?")
                reason = c.get("reason", "")
                matched = c.get("matched_skills", [])
                text += f"\n{i}. {name}（{hours}学时）"
                if matched:
                    text += f" — 覆盖技能：{', '.join(matched)}"
                if reason:
                    text += f" — {reason}"

        return text

    def _get_format_prompt(self) -> str:
        """根据角色返回格式化提示词"""
        if self.role == "employee":
            return """你是一位温暖的职业转型顾问。请根据匹配分析结果，给员工一个专业、鼓励的回复。

要求：
1. 先共情：认可员工转型的勇气和决心
2. 强调优势：明确指出已具备的可迁移技能，让员工看到希望
3. 正视差距：客观说明需要补齐的技能，但用积极语言
4. 给出路径：推荐具体的培训课程和学习步骤
5. 鼓励前行：用积极语言激励员工，强调转型可行性

格式要求：使用清晰的段落和emoji图标，让回复易读。"""
        else:
            return """你是一位资深HR分析专家。请根据匹配分析结果，给出专业的人力资源建议。

要求：
1. 数据概览：匹配度得分和差距等级
2. 技能盘点：已具备技能和缺失技能
3. 培训规划：推荐课程及学时估算
4. 风险提示：转型过程中的关键风险点
5. 行动建议：可落地的下一步行动

格式要求：使用清晰的段落和emoji图标，让回复专业且易读。"""

    # ---------- 处理技能差距 ----------
    def _handle_skill_gap(self, message: str) -> str:
        """处理技能差距分析请求"""
        params = self._extract_matching_params(message)

        if not params["employee_skills"] or not params["target_job_skills"]:
            return self._handle_general(
                message + "\n[系统提示：用户想了解技能差距，但信息不够完整，请引导用户描述现有技能和目标岗位]"
            )

        try:
            gap = matching_engine.analyze_skill_gap(
                params["employee_skills"], params["target_job_skills"]
            )
        except Exception as e:
            error_msg = f"技能差距分析出错：{str(e)}"
            self._save_history(message, error_msg)
            return error_msg

        result_text = f"""【技能差距分析】
已具备技能：{', '.join(gap.get('matched_skills', [])) or '无'}
缺失技能：{', '.join(gap.get('missing_skills', [])) or '无'}
差距等级：{gap.get('gap_severity', 'N/A')}
结论：{gap.get('conclusion', 'N/A')}"""

        messages = [SystemMessage(content=self.system_prompt)]
        for h in self.history[-4:]:
            role_msg = HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"])
            messages.append(role_msg)
        messages.append(HumanMessage(content=f"用户问：{message}\n\n分析结果：\n{result_text}"))

        response = self.llm.invoke(messages)
        self._save_history(message, response.content)
        return response.content

    # ---------- 处理培训路径 ----------
    def _handle_training_path(self, message: str) -> str:
        """处理培训路径请求"""
        params = self._extract_matching_params(message)

        if not params.get("target_job_name"):
            return self._handle_general(
                message + "\n[系统提示：用户想了解培训路径，但未明确目标岗位，请引导用户说明目标]"
            )

        # 如果没有缺失技能，先分析差距
        missing_skills = []
        if params["employee_skills"] and params["target_job_skills"]:
            try:
                gap = matching_engine.analyze_skill_gap(
                    params["employee_skills"], params["target_job_skills"]
                )
                missing_skills = gap.get("missing_skills", [])
            except Exception:
                pass

        if not missing_skills:
            missing_skills = ["智能制造基础", "数字化转型"]

        try:
            training = matching_engine.generate_training_path(
                missing_skills, params.get("target_job_name", "目标岗位")
            )
        except Exception as e:
            error_msg = f"培训路径生成出错：{str(e)}"
            self._save_history(message, error_msg)
            return error_msg

        result_text = f"【推荐培训课程（目标：{params.get('target_job_name', '目标岗位')}）】\n"
        total_hours = 0
        for i, c in enumerate(training, 1):
            name = c.get("name", c.get("course_id", f"课程{i}"))
            hours = c.get("hours", 0)
            total_hours += hours
            reason = c.get("reason", "")
            matched = c.get("matched_skills", [])
            result_text += f"{i}. {name}（{hours}学时）"
            if matched:
                result_text += f" — 覆盖：{', '.join(matched)}"
            if reason:
                result_text += f" — {reason}"
            result_text += "\n"
        result_text += f"\n总计：{len(training)}门课程，{total_hours}学时"

        messages = [SystemMessage(content=self.system_prompt)]
        for h in self.history[-4:]:
            role_msg = HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"])
            messages.append(role_msg)
        messages.append(HumanMessage(content=f"用户问：{message}\n\n培训路径：\n{result_text}"))

        response = self.llm.invoke(messages)
        self._save_history(message, response.content)
        return response.content

    # ---------- HR 汇总分析 ----------
    def _handle_hr_aggregate(self, message: str) -> str:
        """处理HR汇总分析请求"""
        messages = [SystemMessage(content=self.system_prompt)]
        for h in self.history[-6:]:
            role_msg = HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"])
            messages.append(role_msg)
        messages.append(HumanMessage(
            content=message + "\n\n[请从HR/管理层的角度，给出团队转型分析、人力优化建议和培训资源规划。]"
        ))

        response = self.llm.invoke(messages)
        self._save_history(message, response.content)
        return response.content

    # ---------- 通用对话 ----------
    def _handle_general(self, message: str) -> str:
        """通用对话处理"""
        messages = [SystemMessage(content=self.system_prompt)]
        # 加入对话历史
        for h in self.history[-10:]:  # 最近5轮
            role_msg = HumanMessage(content=h["content"]) if h["role"] == "user" else AIMessage(content=h["content"])
            messages.append(role_msg)
        messages.append(HumanMessage(content=message))

        response = self.llm.invoke(messages)
        self._save_history(message, response.content)
        return response.content

    # ---------- 辅助方法 ----------
    def _save_history(self, user_msg: str, assistant_msg: str):
        """保存对话历史"""
        self.history.append({"role": "user", "content": user_msg})
        self.history.append({"role": "assistant", "content": assistant_msg})

    def reset(self):
        """重置对话历史"""
        self.history = []

    def get_history(self) -> List[Dict[str, str]]:
        """获取对话历史"""
        return self.history


# ==================== 会话管理器 ====================

class AgentSessionManager:
    """多会话管理器：为不同用户/会话维护独立的 Agent 实例"""

    def __init__(self):
        self._sessions: Dict[str, SmartMfgAgent] = {}

    def get_agent(self, session_id: str, role: str = "employee") -> SmartMfgAgent:
        """获取或创建指定会话的 Agent"""
        key = f"{role}:{session_id}"
        if key not in self._sessions:
            self._sessions[key] = SmartMfgAgent(role=role, session_id=session_id)
        return self._sessions[key]

    def reset_session(self, session_id: str, role: str = "employee"):
        """重置指定会话"""
        key = f"{role}:{session_id}"
        if key in self._sessions:
            self._sessions[key].reset()

    def list_sessions(self) -> List[str]:
        """列出所有活跃会话"""
        return list(self._sessions.keys())


# 全局会话管理器实例
agent_manager = AgentSessionManager()


# ==================== 快速测试 ====================

if __name__ == "__main__":
    print("=" * 60)
    print("智能制造转型 AI Agent — 快速测试")
    print("=" * 60)

    # 员工端测试
    print("\n【员工端测试】")
    emp_agent = SmartMfgAgent(role="employee", session_id="test-emp")

    test_messages = [
        "你好，我是一名有10年经验的质检员，想转行做AI视觉检测员，你能帮我分析一下吗？",
    ]

    for msg in test_messages:
        print(f"\n用户：{msg}")
        reply = emp_agent.chat(msg)
        print(f"助手：{reply}")

    # HR端测试
    print("\n" + "=" * 60)
    print("【HR端测试】")
    hr_agent = SmartMfgAgent(role="hr", session_id="test-hr")

    hr_msg = "你好，我是HR，想了解一下质检部门员工的转岗情况汇总"
    print(f"\n用户：{hr_msg}")
    reply = hr_agent.chat(hr_msg)
    print(f"助手：{reply}")