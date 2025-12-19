"""
유사도 계산 모듈
Phase 1+2: 의미(Semantic) + 관계(Relational) + 형태(Formative)
"""

import logging
from typing import Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import jamo
import re

logger = logging.getLogger(__name__)


class SimilarityCalculator:
    """유사도 계산 클래스"""
    
    def __init__(self, semantic_model_name: str, nli_model_name: str):
        """
        Args:
            semantic_model_name: 의미 임베딩 모델 이름
            nli_model_name: NLI 모델 이름
        """
        logger.info("모델 로딩 시작...")
        
        # 의미 유사도 모델 (문장 임베딩)
        self.semantic_model = SentenceTransformer(semantic_model_name)
        logger.info(f"✅ Semantic 모델 로딩 완료: {semantic_model_name}")
        
        # 관계 유사도 모델 (NLI)
        self.nli_pipeline = pipeline(
            "text-classification",
            model=nli_model_name,
            device=-1  # CPU 사용 (GPU: 0)
        )
        logger.info(f"✅ NLI 모델 로딩 완료: {nli_model_name}")
        
        logger.info("🚀 모든 모델 로딩 완료!")
    
    def normalize_text(self, text: str) -> str:
        """텍스트 정규화"""
        # 소문자 변환
        text = text.lower()
        # 공백 정리
        text = re.sub(r'\s+', '', text)
        # 특수문자 제거
        text = re.sub(r'[^\w\s가-힣]', '', text)
        return text.strip()
    
    def calculate_semantic_similarity(self, input_text: str, answer: str) -> float:
        """
        의미 유사도 계산 (코사인 유사도)
        
        Args:
            input_text: 사용자 입력
            answer: 정답
            
        Returns:
            0.0 ~ 1.0 사이의 유사도
        """
        # 임베딩 생성
        embeddings = self.semantic_model.encode([input_text, answer])
        
        # 코사인 유사도 계산
        cosine_sim = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        
        # -1 ~ 1 범위를 0 ~ 1로 변환
        similarity = (cosine_sim + 1) / 2
        
        return float(similarity)
    
    def calculate_relational_similarity(
        self, 
        input_text: str, 
        answer: str,
        templates: list,
        contradiction_templates: list
    ) -> Tuple[float, float]:
        """
        관계 유사도 계산 (NLI 기반)
        
        Args:
            input_text: 사용자 입력
            answer: 정답
            templates: 긍정 템플릿 리스트
            contradiction_templates: 반의 템플릿 리스트
            
        Returns:
            (관계 점수, 반의어 점수) 튜플
        """
        entailment_scores = []
        
        # 긍정 템플릿으로 관계 점수 계산
        for template in templates:
            hypothesis = template.format(input=input_text, answer=answer)
            
            result = self.nli_pipeline(hypothesis)[0]
            
            # entailment 확률 추출
            if result['label'] == 'entailment':
                entailment_scores.append(result['score'])
            elif result['label'] == 'neutral':
                entailment_scores.append(result['score'] * 0.5)
            else:  # contradiction
                entailment_scores.append(0.0)
        
        # 평균 관계 점수
        relation_score = np.mean(entailment_scores) if entailment_scores else 0.0
        
        # 반의어 점수 계산
        contradiction_scores = []
        for template in contradiction_templates:
            hypothesis = template.format(input=input_text, answer=answer)
            
            result = self.nli_pipeline(hypothesis)[0]
            
            if result['label'] == 'entailment':
                contradiction_scores.append(result['score'])
        
        # 최대 반의어 점수
        contradiction_score = max(contradiction_scores) if contradiction_scores else 0.0
        
        return float(relation_score), float(contradiction_score)
    
    def analyze_relationship_type(
        self,
        input_text: str,
        answer: str,
        relationship_templates: Dict[str, str]
    ) -> Tuple[str, float]:
        """
        입력과 정답 간 관계 유형 분석
        
        Args:
            input_text: 사용자 입력
            answer: 정답
            relationship_templates: 관계 분석 템플릿 딕셔너리
            
        Returns:
            (관계 유형, 확신도) 튜플
            예: ("사람관계", 0.85)
        """
        relationship_scores = {}
        
        for rel_type, template in relationship_templates.items():
            hypothesis = template.format(input=input_text, answer=answer)
            
            try:
                result = self.nli_pipeline(hypothesis)[0]
                
                # entailment 확률만 사용
                if result['label'] == 'entailment':
                    relationship_scores[rel_type] = result['score']
                else:
                    relationship_scores[rel_type] = 0.0
                    
            except Exception as e:
                logger.warning(f"관계 분석 오류 ({rel_type}): {e}")
                relationship_scores[rel_type] = 0.0
        
        # 가장 높은 점수의 관계 유형 반환
        if relationship_scores:
            best_relationship = max(relationship_scores.items(), key=lambda x: x[1])
            return best_relationship[0], best_relationship[1]
        
        return "일반", 0.0
    
    def calculate_formative_similarity(self, input_text: str, answer: str) -> float:
        """
        형태 유사도 계산 (자모 분해 편집거리)
        
        Args:
            input_text: 사용자 입력
            answer: 정답
            
        Returns:
            0.0 ~ 1.0 사이의 유사도
        """
        # 한글 자모 분해
        input_jamo = jamo.h2j(input_text)
        answer_jamo = jamo.h2j(answer)
        
        # 레벤슈타인 거리 계산
        distance = self._levenshtein_distance(input_jamo, answer_jamo)
        
        # 최대 길이로 정규화
        max_len = max(len(input_jamo), len(answer_jamo))
        if max_len == 0:
            return 1.0
        
        similarity = 1.0 - (distance / max_len)
        
        return float(max(0.0, similarity))
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """레벤슈타인 편집거리 계산"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # 삽입, 삭제, 치환 비용
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def calculate_combined_similarity(
        self,
        input_text: str,
        answer: str,
        weights: Dict[str, float],
        nli_templates: list,
        contradiction_templates: list
    ) -> Dict[str, float]:
        """
        종합 유사도 계산
        
        Args:
            input_text: 사용자 입력
            answer: 정답
            weights: 가중치 딕셔너리
            nli_templates: NLI 긍정 템플릿
            contradiction_templates: NLI 반의 템플릿
            
        Returns:
            {
                "similarity_score": 최종 점수 (0-100),
                "breakdown": {
                    "semantic": 의미 점수,
                    "relational": 관계 점수,
                    "formative": 형태 점수,
                    "contradiction": 반의어 감점
                }
            }
        """
        # 텍스트 정규화
        input_normalized = self.normalize_text(input_text)
        answer_normalized = self.normalize_text(answer)
        
        # 정답 체크
        if input_normalized == answer_normalized:
            return {
                "similarity_score": 100.0,
                "breakdown": {
                    "semantic": 1.0,
                    "relational": 1.0,
                    "formative": 1.0,
                    "contradiction": 0.0
                }
            }
        
        # 각 유사도 계산
        semantic_score = self.calculate_semantic_similarity(input_text, answer)
        relation_score, contradiction_score = self.calculate_relational_similarity(
            input_text, answer, nli_templates, contradiction_templates
        )
        formative_score = self.calculate_formative_similarity(input_normalized, answer_normalized)
        
        # 가중 평균 계산
        weighted_score = (
            semantic_score * weights["semantic"] +
            relation_score * weights["relational"] +
            formative_score * weights["formative"]
        )
        
        # 반의어 감점 적용
        weighted_score = weighted_score - (contradiction_score * 0.15)
        
        # 0~1 범위로 클리핑
        weighted_score = max(0.0, min(1.0, weighted_score))
        
        # 백분율로 변환
        final_score = weighted_score * 100
        
        return {
            "similarity_score": round(final_score, 2),
            "breakdown": {
                "semantic": round(semantic_score, 4),
                "relational": round(relation_score, 4),
                "formative": round(formative_score, 4),
                "contradiction": round(contradiction_score, 4)
            }
        }