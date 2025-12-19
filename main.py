"""
WordTreasure 유사도 계산 서비스
FastAPI 서버
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from models.similarity import SimilarityCalculator
from models.hint import HintGenerator
import config

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 전역 변수로 모델 저장
similarity_calculator = None
hint_generator = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """애플리케이션 시작/종료 시 실행"""
    global similarity_calculator, hint_generator
    
    logger.info("=" * 50)
    logger.info("🚀 WordTreasure 유사도 서비스 시작")
    logger.info("=" * 50)
    
    # 모델 로딩
    logger.info("📦 모델 로딩 시작... (약 30초 소요)")
    start_time = time.time()
    
    try:
        similarity_calculator = SimilarityCalculator(
            semantic_model_name=config.SEMANTIC_MODEL,
            nli_model_name=config.NLI_MODEL
        )
        hint_generator = HintGenerator(
            hint_thresholds=config.HINT_THRESHOLDS,
            contextual_templates=config.CONTEXTUAL_HINT_TEMPLATES,
            detail_suffixes=config.DETAIL_HINT_SUFFIX
        )
        
        elapsed = time.time() - start_time
        logger.info(f"✅ 모델 로딩 완료! (소요 시간: {elapsed:.2f}초)")
        
        # 자동 워밍업
        logger.info("🔥 워밍업 시작... (모델 캐시 최적화)")
        warmup_start = time.time()
        
        try:
            # 더미 요청으로 모델 워밍업
            warmup_result = similarity_calculator.calculate_combined_similarity(
                input_text="워밍업",
                answer="테스트",
                weights=config.WEIGHTS,
                nli_templates=config.NLI_TEMPLATES,
                contradiction_templates=config.CONTRADICTION_TEMPLATES
            )
            
            # 관계 분석도 워밍업
            similarity_calculator.analyze_relationship_type(
                input_text="워밍업",
                answer="테스트",
                relationship_templates=config.RELATIONSHIP_ANALYSIS_TEMPLATES
            )
            
            warmup_elapsed = time.time() - warmup_start
            logger.info(f"✅ 워밍업 완료! (소요 시간: {warmup_elapsed:.2f}초)")
            logger.info(f"⚡ 이제 모든 요청이 빠르게 처리됩니다! (예상: ~150ms)")
            
        except Exception as e:
            logger.warning(f"⚠️ 워밍업 중 오류 (무시 가능): {e}")
        
        logger.info(f"🌐 서버 준비 완료: http://{config.SERVER_HOST}:{config.SERVER_PORT}")
        logger.info(f"💡 맥락적 힌트 시스템 활성화!")
        logger.info("=" * 50)
        
    except Exception as e:
        logger.error(f"❌ 모델 로딩 실패: {e}")
        raise
    
    yield
    
    logger.info("👋 서버 종료")


# FastAPI 앱 생성
app = FastAPI(
    title="WordTreasure Similarity Service",
    description="AI 기반 단어 유사도 측정 및 힌트 생성 서비스",
    version="1.0.0 (Phase 1+2)",
    lifespan=lifespan
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 요청/응답 모델
class SimilarityRequest(BaseModel):
    """유사도 계산 요청"""
    user_input: str = Field(..., description="사용자가 입력한 단어", min_length=1)
    answer: str = Field(..., description="정답 단어", min_length=1)
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_input": "친구",
                "answer": "배신"
            }
        }


class SimilarityResponse(BaseModel):
    """유사도 계산 응답"""
    similarity_score: float = Field(..., description="유사도 점수 (0-100)")
    hint: str = Field(..., description="생성된 힌트")
    category_match: bool = Field(..., description="카테고리 매칭 여부")
    breakdown: Dict[str, float] = Field(..., description="세부 점수")
    processing_time_ms: float = Field(..., description="처리 시간 (밀리초)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "similarity_score": 45.32,
                "hint": "단어의 뜻은 비슷하지만 관계나 맥락이 조금 달라요.",
                "category_match": False,
                "breakdown": {
                    "semantic": 0.62,
                    "relational": 0.38,
                    "formative": 0.15,
                    "contradiction": 0.05
                },
                "processing_time_ms": 123.45
            }
        }


# API 엔드포인트
@app.get("/")
async def root():
    """서비스 상태 확인"""
    return {
        "service": "WordTreasure Similarity Service",
        "version": "1.0.0 (Phase 1+2 Enhanced)",
        "status": "running",
        "features": [
            "Semantic Similarity (의미 유사도)",
            "Relational Similarity (관계 유사도)",
            "Formative Similarity (형태 유사도)",
            "Contextual Hints (맥락적 힌트) ✨ NEW",
            "Relationship Analysis (관계 분석) ✨ NEW"
        ]
    }


@app.get("/health")
async def health_check():
    """헬스 체크"""
    return {
        "status": "healthy",
        "models_loaded": similarity_calculator is not None and hint_generator is not None
    }


@app.post("/api/similarity/calculate", response_model=SimilarityResponse)
async def calculate_similarity(request: SimilarityRequest):
    """
    유사도 계산 및 힌트 생성
    
    Args:
        request: 사용자 입력 및 정답
        
    Returns:
        유사도 점수, 힌트, 세부 분석
    """
    if similarity_calculator is None or hint_generator is None:
        raise HTTPException(status_code=503, detail="모델이 아직 로딩되지 않았습니다.")
    
    start_time = time.time()
    
    try:
        logger.info(f"📝 요청 - 입력: '{request.user_input}', 정답: '{request.answer}'")

        # 입력과 정답이 완전히 동일한 경우(정규화 기준):
        # 공백/대소문자/특수문자 차이는 무시하고 같으면 유사도 계산/관계 분석을 생략하고 바로 100% 반환
        normalized_input = similarity_calculator.normalize_text(request.user_input)
        normalized_answer = similarity_calculator.normalize_text(request.answer)
        if normalized_input == normalized_answer:
            processing_time = (time.time() - start_time) * 1000  # ms
            logger.info(
                f"✅ 완전 일치 - 유사도: 100.0%, 처리 시간: {processing_time:.2f}ms "
                "(모델 호출 생략)"
            )
            return SimilarityResponse(
                similarity_score=100.0,
                hint="정답과 완전히 동일한 단어예요!",
                category_match=True,
                breakdown={
                    "semantic": 1.0,
                    "relational": 1.0,
                    "formative": 1.0,
                    "contradiction": 0.0,
                },
                processing_time_ms=round(processing_time, 2),
            )
        
        # 유사도 계산
        result = similarity_calculator.calculate_combined_similarity(
            input_text=request.user_input,
            answer=request.answer,
            weights=config.WEIGHTS,
            nli_templates=config.NLI_TEMPLATES,
            contradiction_templates=config.CONTRADICTION_TEMPLATES
        )
        
        # 관계 유형 분석 (맥락적 힌트용)
        relationship_type, relationship_confidence = similarity_calculator.analyze_relationship_type(
            input_text=request.user_input,
            answer=request.answer,
            relationship_templates=config.RELATIONSHIP_ANALYSIS_TEMPLATES
        )
        
        logger.info(f"🔍 관계 분석 - 유형: '{relationship_type}', 확신도: {relationship_confidence:.2f}")
        
        # 힌트 생성 (관계 정보 포함)
        hint = hint_generator.generate_hint(
            similarity_score=result["similarity_score"],
            breakdown=result["breakdown"],
            user_input=request.user_input,
            answer=request.answer,
            relationship_type=relationship_type,
            relationship_confidence=relationship_confidence
        )
        
        # 카테고리 매칭 (Phase 3에서 구현 예정)
        category_info = hint_generator.generate_category_hint(
            user_input=request.user_input,
            answer=request.answer
        )
        
        # 처리 시간 계산
        processing_time = (time.time() - start_time) * 1000  # ms
        
        logger.info(f"✅ 응답 - 유사도: {result['similarity_score']}%, 힌트: '{hint}', 처리 시간: {processing_time:.2f}ms")
        
        return SimilarityResponse(
            similarity_score=result["similarity_score"],
            hint=hint,
            category_match=category_info["category_match"],
            breakdown=result["breakdown"],
            processing_time_ms=round(processing_time, 2)
        )
        
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"유사도 계산 중 오류 발생: {str(e)}")


@app.get("/api/config")
async def get_config():
    """현재 설정 조회"""
    return {
        "weights": config.WEIGHTS,
        "models": {
            "semantic": config.SEMANTIC_MODEL,
            "nli": config.NLI_MODEL
        },
        "target_latency_ms": config.TARGET_LATENCY_MS
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "main:app",
        host=config.SERVER_HOST,
        port=config.SERVER_PORT,
        reload=False,  # 프로덕션에서는 False
        log_level="info"
    )