"""
技能差距分析模块
对比员工现有技能与目标岗位要求，计算差距并生成分析报告
"""
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import json


@dataclass
class SkillGap:
    """单项技能差距"""
    skill_name: str
    current_level: float  # 员工当前水平 0-1
    required_level: float  # 岗位要求水平 0-1
    gap: float  # 差距值 = required_level - current_level
    importance_weight: float  # 技能重要性权重
    is_mandatory: bool  # 是否必要技能
    gap_severity: str = ""  # 差距严重程度：none/minor/moderate/critical

    def __post_init__(self):
        if self.gap <= 0:
            self.gap_severity = "none"
        elif self.gap <= 0.2:
            self.gap_severity = "minor"
        elif self.gap <= 0.5:
            self.gap_severity = "moderate"
        else:
            self.gap_severity = "critical"


@dataclass
class GapAnalysisResult:
    """技能差距分析结果"""
    employee_id: str
    position_id: str
    skill_gaps: List[SkillGap] = field(default_factory=list)
    overall_gap_score: float = 0.0  # 综合差距分数 0-100（越低越好）
    matched_skills: List[str] = field(default_factory=list)
    gap_skills: List[str] = field(default_factory=list)
    critical_gaps: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            "employee_id": self.employee_id,
            "position_id": self.position_id,
            "overall_gap_score": round(self.overall_gap_score, 2),
            "matched_skills": self.matched_skills,
            "gap_skills": self.gap_skills,
            "critical_gaps": self.critical_gaps,
            "skill_gaps": [
                {
                    "skill_name": g.skill_name,
                    "current_level": round(g.current_level, 2),
                    "required_level": round(g.required_level, 2),
                    "gap": round(g.gap, 2),
                    "gap_severity": g.gap_severity,
                    "is_mandatory": g.is_mandatory,
                }
                for g in self.skill_gaps
            ],
            "summary": self.summary,
        }


class SkillGapAnalyzer:
    """技能差距分析器"""

    def __init__(self, gap_threshold: float = 0.3):
        """
        Args:
            gap_threshold: 差距阈值，超过此值视为存在差距
        """
        self.gap_threshold = gap_threshold

    def analyze(
        self,
        employee_skills: List[Dict],
        position_skills: List[Dict],
        employee_id: str = "",
        position_id: str = "",
    ) -> GapAnalysisResult:
        """
        执行技能差距分析
        Args:
            employee_skills: 员工技能列表 [{"name": str, "proficiency_level": float}]
            position_skills: 岗位技能要求 [{"name": str, "required_level": float, "importance_weight": float, "is_mandatory": bool}]
            employee_id: 员工ID
            position_id: 岗位ID
        Returns:
            GapAnalysisResult 分析结果
        """
        # 构建员工技能映射
        emp_skill_map = {
            s["name"]: s.get("proficiency_level", 0.0) for s in employee_skills
        }

        result = GapAnalysisResult(
            employee_id=employee_id,
            position_id=position_id,
        )

        weighted_gap_sum = 0.0
        total_weight = 0.0

        for pos_skill in position_skills:
            skill_name = pos_skill["name"]
            required_level = pos_skill.get("required_level", 0.0)
            weight = pos_skill.get("importance_weight", 1.0)
            is_mandatory = bool(pos_skill.get("is_mandatory", 1))

            current_level = emp_skill_map.get(skill_name, 0.0)
            gap = required_level - current_level

            skill_gap = SkillGap(
                skill_name=skill_name,
                current_level=current_level,
                required_level=required_level,
                gap=max(0, gap),
                importance_weight=weight,
                is_mandatory=is_mandatory,
            )
            result.skill_gaps.append(skill_gap)

            # 计算加权差距
            weighted_gap_sum += max(0, gap) * weight
            total_weight += weight

            # 分类
            if gap <= 0:
                result.matched_skills.append(skill_name)
            elif skill_gap.gap_severity == "critical":
                result.critical_gaps.append(skill_name)
                result.gap_skills.append(skill_name)
            else:
                result.gap_skills.append(skill_name)

        # 计算综合差距分数（0-100，0=完美匹配，100=完全不匹配）
        if total_weight > 0:
            normalized_gap = weighted_gap_sum / total_weight
            result.overall_gap_score = min(100, normalized_gap * 100)
        else:
            result.overall_gap_score = 0.0

        # 生成摘要
        result.summary = self._generate_summary(result)

        return result

    def _generate_summary(self, result: GapAnalysisResult) -> str:
        """生成分析摘要文本"""
        total = len(result.skill_gaps)
        matched = len(result.matched_skills)
        gaps = len(result.gap_skills)
        critical = len(result.critical_gaps)

        summary_parts = [
            f"技能差距分析：共{total}项技能要求，已匹配{matched}项，存在差距{gaps}项",
        ]

        if critical > 0:
            summary_parts.append(f"其中{critical}项为关键差距：{', '.join(result.critical_gaps)}")

        if result.overall_gap_score <= 20:
            summary_parts.append("整体匹配度较高，技能差距较小。")
        elif result.overall_gap_score <= 50:
            summary_parts.append("存在一定技能差距，建议针对性培训提升。")
        else:
            summary_parts.append("技能差距较大，需要系统性技能重塑计划。")

        return "。".join(summary_parts)