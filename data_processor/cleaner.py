"""
数据清洗模块
负责对公开数据源（岗位描述、员工简历等）进行标准化清洗
"""
import re
import json
from typing import List, Dict, Optional
from datetime import datetime


class DataCleaner:
    """数据清洗器，处理原始数据的标准化"""

    def __init__(self):
        # 智能制造相关技能关键词映射（用于技能标准化）
        self.skill_alias_map = {
            "PLC编程": ["PLC", "可编程逻辑控制器", "PLC编程"],
            "SCADA": ["SCADA", "数据采集与监视控制系统"],
            "MES": ["MES", "制造执行系统"],
            "工业机器人": ["工业机器人", "机器人编程", "Robot Programming"],
            "Python": ["Python", "python", "PYTHON"],
            "数字孪生": ["数字孪生", "Digital Twin"],
            "工业物联网": ["IIoT", "工业物联网", "Industrial IoT"],
            "机器视觉": ["机器视觉", "Machine Vision", "视觉检测"],
            "CNC加工": ["CNC", "数控加工", "数控机床"],
            "ERP": ["ERP", "企业资源计划"],
        }

    def clean_text(self, text: str) -> str:
        """基础文本清洗：去除多余空白、特殊字符等"""
        if not text:
            return ""
        # 去除 HTML 标签
        text = re.sub(r"<[^>]+>", "", text)
        # 去除多余空白
        text = re.sub(r"\s+", " ", text).strip()
        # 去除不可见字符
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        return text

    def normalize_skill_name(self, skill_name: str) -> str:
        """将技能别名标准化为统一名称"""
        skill_name = self.clean_text(skill_name)
        for standard_name, aliases in self.skill_alias_map.items():
            if skill_name in aliases:
                return standard_name
        return skill_name

    def clean_employee_data(self, raw_data: Dict) -> Dict:
        """
        清洗员工数据
        Args:
            raw_data: 原始员工数据字典
        Returns:
            清洗后的标准化员工数据
        """
        cleaned = {}
        cleaned["name"] = self.clean_text(raw_data.get("name", ""))
        cleaned["employee_id"] = self.clean_text(raw_data.get("employee_id", ""))
        cleaned["department"] = self.clean_text(raw_data.get("department", ""))
        cleaned["current_position"] = self.clean_text(raw_data.get("current_position", ""))
        cleaned["years_of_service"] = float(raw_data.get("years_of_service", 0) or 0)
        cleaned["education"] = self.clean_text(raw_data.get("education", ""))
        cleaned["age"] = int(raw_data.get("age", 0) or 0)
        cleaned["resume_text"] = self.clean_text(raw_data.get("resume_text", ""))

        # 标准化技能列表
        raw_skills = raw_data.get("skills", [])
        cleaned["skills"] = []
        for skill in raw_skills:
            normalized = {
                "name": self.normalize_skill_name(skill.get("name", "")),
                "proficiency_level": min(1.0, max(0.0, float(skill.get("proficiency_level", 0)))),
                "years_of_experience": float(skill.get("years_of_experience", 0) or 0),
            }
            cleaned["skills"].append(normalized)

        return cleaned

    def clean_position_data(self, raw_data: Dict) -> Dict:
        """
        清洗岗位数据
        Args:
            raw_data: 原始岗位数据字典
        Returns:
            清洗后的标准化岗位数据
        """
        cleaned = {}
        cleaned["title"] = self.clean_text(raw_data.get("title", ""))
        cleaned["department"] = self.clean_text(raw_data.get("department", ""))
        cleaned["level"] = self.clean_text(raw_data.get("level", "中级"))
        cleaned["description"] = self.clean_text(raw_data.get("description", ""))
        cleaned["responsibility"] = self.clean_text(raw_data.get("responsibility", ""))
        cleaned["is_intelligent_mfg"] = int(raw_data.get("is_intelligent_mfg", 0))

        # 标准化技能要求
        raw_skills = raw_data.get("skills", [])
        cleaned["skills"] = []
        for skill in raw_skills:
            normalized = {
                "name": self.normalize_skill_name(skill.get("name", "")),
                "required_level": min(1.0, max(0.0, float(skill.get("required_level", 0)))),
                "importance_weight": min(2.0, max(0.1, float(skill.get("importance_weight", 1.0)))),
                "is_mandatory": int(skill.get("is_mandatory", 1)),
            }
            cleaned["skills"].append(normalized)

        return cleaned

    def batch_clean(self, data_list: List[Dict], data_type: str = "employee") -> List[Dict]:
        """
        批量清洗数据
        Args:
            data_list: 原始数据列表
            data_type: 数据类型 employee/position
        Returns:
            清洗后的数据列表
        """
        clean_func = self.clean_employee_data if data_type == "employee" else self.clean_position_data
        return [clean_func(item) for item in data_list if item]