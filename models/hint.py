"""
힌트 생성 모듈
사용자 입력과 유사도를 기반으로 맞춤형 힌트 제공
Phase 1+2: 맥락적 관계 분석 기반 힌트
"""

from typing import Dict


class HintGenerator:
    """힌트 생성 클래스 - 맥락적 힌트 시스템"""
    
    def __init__(
        self, 
        hint_thresholds: Dict[int, str],
        contextual_templates: Dict[str, Dict[str, str]] = None,
        detail_suffixes: Dict[str, str] = None
    ):
        """
        Args:
            hint_thresholds: 점수별 기본 힌트 매핑
            contextual_templates: 관계 유형별 맥락적 힌트 템플릿
            detail_suffixes: 세부 점수 기반 추가 힌트
        """
        self.hint_thresholds = hint_thresholds
        self.contextual_templates = contextual_templates or {}
        self.detail_suffixes = detail_suffixes or {}
    
    def generate_hint(
        self, 
        similarity_score: float, 
        breakdown: Dict[str, float],
        user_input: str,
        answer: str,
        relationship_type: str = None,
        relationship_confidence: float = 0.0
    ) -> str:
        """
        유사도 + 관계 분석 기반 맥락적 힌트 생성
        
        Args:
            similarity_score: 최종 유사도 점수 (0-100)
            breakdown: 세부 점수 (semantic, relational, formative)
            user_input: 사용자 입력
            answer: 정답
            relationship_type: 분석된 관계 유형 (예: "사람관계")
            relationship_confidence: 관계 분석 확신도 (0-1)
            
        Returns:
            생성된 힌트 문자열
        """
        # 정답인 경우
        if similarity_score == 100.0:
            return "정답입니다! 🎉"
        
        # 1. 맥락적 힌트 생성 시도
        contextual_hint = self._generate_contextual_hint(
            user_input, 
            similarity_score,
            relationship_type, 
            relationship_confidence
        )
        
        # 2. 세부 분석 기반 추가 힌트
        detail_hint = self._get_detail_hint(breakdown, similarity_score)
        
        # 3. 힌트 조합
        if contextual_hint:
            # 맥락적 힌트가 있으면 우선 사용
            if detail_hint and similarity_score >= 60:
                # 점수가 높으면 세부 힌트도 추가
                return f"{contextual_hint}. {detail_hint}"
            return contextual_hint
        else:
            # 맥락적 힌트 생성 실패 시 기본 힌트 사용
            base_hint = self._get_base_hint(similarity_score)
            if detail_hint:
                return f"{base_hint} {detail_hint}"
            return base_hint
    
    def _generate_contextual_hint(
        self,
        user_input: str,
        similarity_score: float,
        relationship_type: str,
        relationship_confidence: float
    ) -> str:
        """
        관계 유형 기반 맥락적 힌트 생성
        
        Args:
            user_input: 사용자 입력
            similarity_score: 유사도 점수
            relationship_type: 관계 유형
            relationship_confidence: 확신도
            
        Returns:
            맥락적 힌트 또는 빈 문자열
        """
        # 관계 분석이 없거나 확신도가 낮으면 스킵
        if not relationship_type or relationship_confidence < 0.3:
            return ""
        
        # 점수가 너무 낮으면 맥락적 힌트 제공 안 함
        if similarity_score < 15:
            return ""
        
        # 관계 유형에 해당하는 템플릿 가져오기
        templates = self.contextual_templates.get(relationship_type, {})
        if not templates:
            return ""
        
        # 확신도와 점수에 따라 힌트 레벨 결정
        if relationship_confidence >= 0.7 and similarity_score >= 50:
            hint_level = "high"
        elif relationship_confidence >= 0.5 or similarity_score >= 30:
            hint_level = "medium"
        else:
            hint_level = "low"
        
        # 템플릿에서 힌트 생성
        hint_template = templates.get(hint_level, templates.get("medium", ""))
        if not hint_template:
            return ""
        
        # {input} 부분을 실제 사용자 입력으로 치환
        hint = hint_template.replace("{input}", user_input)
        
        return hint
    
    def _get_base_hint(self, score: float) -> str:
        """점수 구간별 기본 힌트"""
        for threshold in sorted(self.hint_thresholds.keys(), reverse=True):
            if score >= threshold:
                return self.hint_thresholds[threshold]
        
        return self.hint_thresholds[0]
    
    def _get_detail_hint(self, breakdown: Dict[str, float], score: float) -> str:
        """세부 점수 분석 기반 추가 힌트"""
        semantic = breakdown.get("semantic", 0)
        relational = breakdown.get("relational", 0)
        formative = breakdown.get("formative", 0)
        contradiction = breakdown.get("contradiction", 0)
        
        # 반의어 감지
        if contradiction > 0.6:
            return "하지만 반대 의미는 아니에요"
        
        # 점수가 너무 낮으면 구체적 힌트 제공 안 함
        if score < 20:
            return ""
        
        # 가장 높은 점수 영역 찾기
        max_component = max(
            [("semantic", semantic), ("relational", relational), ("formative", formative)],
            key=lambda x: x[1]
        )
        
        component_name, component_score = max_component
        
        # 각 영역별 힌트
        if component_name == "semantic" and semantic > 0.6:
            if relational < 0.3:
                return self.detail_suffixes.get("semantic_high", "의미적으로 가까워요")
            return ""
        
        elif component_name == "relational" and relational > 0.6:
            if semantic < 0.3:
                return self.detail_suffixes.get("relational_high", "상황이나 맥락은 맞아요")
            return ""
        
        elif component_name == "formative" and formative > 0.7:
            return self.detail_suffixes.get("formative_high", "철자가 매우 비슷해요")
        
        return ""
    
    def generate_category_hint(
        self,
        user_input: str,
        answer: str,
        domain: str = None
    ) -> Dict[str, bool]:
        """
        카테고리 매칭 정보
        
        Args:
            user_input: 사용자 입력
            answer: 정답
            domain: 정답의 도메인 (예: "game", "emotion", "food")
            
        Returns:
            {"category_match": True/False}
        """
        # Phase 1+2에서는 간단하게 구현
        # Phase 3에서 도메인 게이팅 추가 예정
        
        return {
            "category_match": False  # 추후 구현
        }