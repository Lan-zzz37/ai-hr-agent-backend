"""
岗位匹配引擎
综合技能匹配、经验匹配和语义相似度，计算人岗匹配分数
"""
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from backend.matching_engine.skill_gap import SkillGapAnalyzer, GapAnalysisResult
from backend.database.chroma_client import chroma_manager
from backend.config import MATCHING_TOP_K


@dataclass
class MatchScore:
    """单项匹配结果"""
    employee_id: str
    position_id: str
    position_title: str
    overall_score: float  # 综合匹配分数 0-100
    skill_match_score: float  # 技能匹配分数
    experience_score: float  # 经验匹配分数
    semantic_score: float  # 语义相似度分数
    gap_analysis: Optional[GapAnalysisResult] = None

    def to_dict(self) -> Dict:
        return {
            "employee_id": self.employee_id,
            "position_id": self.position_id,
            "position_title": self.position_title,
            "overall_score": round(self.overall_score, 2),
            "skill_match_score": round(self.skill_match_score, 2),
            "experience_score": round(self.experience_score, 2),
            "semantic_score": round(self.semantic_score, 2),
            "gap_analysis": self.gap_analysis.to_dict() if self.gap_analysis else None,
        }


class PositionMatcher:
    """岗位匹配器"""

    def __init__(self, top_k: int = MATCHING_TOP_K):
        """
        Args:
            top_k: 返回Top-K匹配结果
        """
        self.top_k = top_k
        self.gap_analyzer = SkillGapAnalyzer()

        # 匹配权重配置
        self.weight_skill = 0.5  # 技能匹配权重
        self.weight_experience = 0.2  # 经验匹配权重
        self.weight_semantic = 0.3  # 语义相似度权重

    def match_employee_to_positions(
        self,
        employee: Dict,
        positions: List[Dict],
        use_semantic: bool = True,
    ) -> List[MatchScore]:
        """
        为员工匹配最佳岗位
        Args:
            employee: 员工信息 {"id": str, "skills": [...], "years_of_service": float, ...}
            positions: 候选岗位列表 [{"id": str, "title": str, "skills": [...], ...}]
            use_semantic: 是否使用语义匹配
        Returns:
            按匹配分数排序的匹配结果列表
        """
        results = []

        for position in positions:
            # 1. 技能差距分析
            gap_result = self.gap_analyzer.analyze(
                employee_skills=employee.get("skills", []),
                position_skills=position.get("skills", []),
                employee_id=employee.get("id", ""),
                position_id=position.get("id", ""),
            )

            # 2. 技能匹配分数（基于差距分数反算）
            skill_match_score = max(0, 100 - gap_result.overall_gap_score)

            # 3. 经验匹配分数
            experience_score = self._calc_experience_score(
                employee_years=employee.get("years_of_service", 0),
                position_level=position.get("level", "中级"),
            )

            # 4. 语义相似度分数
            semantic_score = 0.0
            if use_semantic:
                semantic_score = self._calc_semantic_score(
                    employee_id=employee.get("id", ""),
                    position_id=position.get("id", ""),
                )

            # 5. 综合匹配分数
            overall_score = (
                self.weight_skill * skill_match_score
                + self.weight_experience * experience_score
                + self.weight_semantic * semantic_score
            )

            match_score = MatchScore(
                employee_id=employee.get("id", ""),
                position_id=position.get("id", ""),
                position_title=position.get("title", ""),
                overall_score=overall_score,
                skill_match_score=skill_match_score,
                experience_score=experience_score,
                semantic_score=semantic_score,
                gap_analysis=gap_result,
            )
            results.append(match_score)

        # 按综合分数降序排列
        results.sort(key=lambda x: x.overall_score, reverse=True)

        return results[: self.top_k]

    def _calc_experience_score(
        self,
        employee_years: float,
        position_level: str,
    ) -> float:
        """
        计算经验匹配分数
        根据岗位级别要求的工作年限与员工实际年限对比
        """
        level_year_map = {
            "初级": 1,
            "中级": 3,
            "高级": 5,
            "专家": 8,
        }
        required_years = level_year_map.get(position_level, 3)

        if employee_years >= required_years:
            # 超出要求年限，给高分但不过度奖励
            return min(100, 80 + (employee_years - required_years) * 2)
        else:
            # 不足要求年限，按比例计算
            ratio = employee_years / required_years if required_years > 0 else 0
            return ratio * 80

    def _calc_semantic_score(
        self,
        employee_id: str,
        position_id: str,
    ) -> float:
        """
        计算语义相似度分数
        通过 ChromaDB 检索员工简历与岗位描述的语义相似度
        """
        try:
            # 查询员工简历集合
            emp_collection = chroma_manager.get_collection("employee_profiles")
            emp_data = emp_collection.get(
                where={"employee_id": employee_id},
                include=["documents"],
            )

            if not emp_data["documents"]:
                return 50.0  # 无数据时返回中等分数

            # 用员工简历文本查询岗位描述集合
            query_text = emp_data["documents"][0]
            pos_collection = chroma_manager.get_collection("position_descriptions")
            results = pos_collection.query(
                query_texts=[query_text],
                n_results=1,
                where={"position_id": position_id},
            )

            if results["distances"] and results["distances"][0]:
                # cosine distance -> similarity score (0-100)
                distance = results["distances"][0][0]
                similarity = max(0, 1 - distance)
                return similarity * 100

            return 50.0
        except Exception:
            # 向量库未初始化或无数据时返回中等分数
            return 50.0

    def batch_match(
        self,
        employees: List[Dict],
        positions: List[Dict],
    ) -> Dict[str, List[MatchScore]]:
        """
        批量匹配：为多个员工同时匹配岗位
        Args:
            employees: 员工列表
            positions: 岗位列表
        Returns:
            {employee_id: [MatchScore, ...]} 字典
        """
        results = {}
        for emp in employees:
            emp_id = emp.get("id", "")
            results[emp_id] = self.match_employee_to_positions(emp, positions)
        return results