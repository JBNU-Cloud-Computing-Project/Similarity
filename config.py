"""
유사도 계산 시스템 설정
"""

# 서버 설정
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8000

# 모델 설정
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# NLI 모델 - CPU 최적화 + safetensors 지원 버전
# 옵션 1: BART (권장, safetensors 지원)
NLI_MODEL = "facebook/bart-large-mnli"

# 옵션 2: XLM-RoBERTa (다국어, safetensors 지원)
# NLI_MODEL = "joeddav/xlm-roberta-large-xnli"

# 주의: microsoft/deberta-v3-small은 PyTorch 2.6+ 필요
# 또는 safetensors 버전 사용: "microsoft/deberta-v3-small-safetensors"
#NLI_MODEL = "microsoft/deberta-v3-small--safetensors"
#NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"  # 다국어 NLI 모델

# 유사도 가중치 (Phase 1+2)
WEIGHTS = {
    "semantic": 0.50,      # 의미 유사도
    "relational": 0.35,    # 관계 유사도
    "formative": 0.15,     # 형태 유사도
}

# NLI 템플릿 (한국어)
NLI_TEMPLATES = [
    "{input}은 {answer}과 관련이 있다.",
    "{input}는 {answer}와 같은 맥락에서 언급된다.",
    "{input}는 {answer}의 상황에서 나타날 수 있다.",
    "{input}와 {answer}는 비슷한 의미를 가진다.",
]

# 반의어 감점 템플릿
CONTRADICTION_TEMPLATES = [
    "{input}은 {answer}과 반대되는 의미다.",
    "{input}와 {answer}는 서로 상반된다.",
]

# 힌트 생성 임계값
HINT_THRESHOLDS = {
    95: "거의 정답이에요! 더 정확한 표현이 있어요.",
    80: "아주 가까워요! 조금만 더 생각해보세요.",
    60: "비슷한 방향이에요. 더 구체적으로 표현해보세요.",
    40: "관련이 있지만 정확하지 않아요.",
    20: "방향이 조금 다른 것 같아요.",
    0: "전혀 다른 방향이에요. 다시 생각해보세요.",
}

# 응답 시간 목표 (ms)
TARGET_LATENCY_MS = 150

# ==================== 맥락적 힌트 시스템 ====================

# 관계 분석 템플릿 (NLI로 입력-정답 간 관계 파악)
RELATIONSHIP_ANALYSIS_TEMPLATES = {
    "상황발생": "{answer}는 {input} 상황에서 발생할 수 있다.",
    "감정원인": "{input}는 {answer}의 원인이 될 수 있다.",
    "속성관계": "{answer}는 {input}의 특성을 가지고 있다.",
    "장소관계": "{answer}는 {input}에서 일어나는 일이다.",
    "사람관계": "{answer}는 {input} 사이에서 나타나는 것이다.",
    "유사장르": "{answer}는 {input}와 비슷한 종류다.",
    "반대관계": "{answer}는 {input}과 반대되는 것이다.",
    "부분전체": "{answer}는 {input}의 일부분이다.",
    "시간관계": "{answer}는 {input} 때 일어나는 것이다.",
    "결과관계": "{answer}는 {input}의 결과로 생기는 것이다.",
}

# 맥락적 힌트 템플릿 (추상적 스타일)
CONTEXTUAL_HINT_TEMPLATES = {
    "상황발생": {
        "high": "{input} 상황에서 나타나는 것이에요",
        "medium": "{input}와/과 관련된 상황에서 일어나는 일이에요",
        "low": "{input} 맥락과 연결된 개념이에요",
    },
    "감정원인": {
        "high": "{input}에서 비롯되는 감정이나 행동이에요",
        "medium": "{input} 때문에 생길 수 있는 것이에요",
        "low": "{input}와/과 인과관계가 있어요",
    },
    "속성관계": {
        "high": "{input}의 성질을 가진 것이에요",
        "medium": "{input}와/과 비슷한 특징이 있어요",
        "low": "{input} 계열의 것이에요",
    },
    "장소관계": {
        "high": "{input}에서 경험할 수 있는 것이에요",
        "medium": "{input}와/과 관련된 장소에서 일어나는 일이에요",
        "low": "{input} 공간과 연결되어 있어요",
    },
    "사람관계": {
        "high": "{input} 사이에서 나타나는 것이에요",
        "medium": "{input} 관계에서 발생할 수 있는 일이에요",
        "low": "{input}와/과 관련된 인간관계 개념이에요",
    },
    "유사장르": {
        "high": "{input}와/과 비슷한 방식으로 진행되는 것이에요",
        "medium": "{input}와/과 같은 종류에 속해요",
        "low": "{input} 계열의 또 다른 것이에요",
    },
    "반대관계": {
        "high": "{input}과는 반대되는 개념이에요",
        "medium": "{input}의 반대 방향에 있는 것이에요",
        "low": "{input}와/과 대조적인 것이에요",
    },
    "부분전체": {
        "high": "{input}의 한 부분이에요",
        "medium": "{input}를/을 구성하는 요소예요",
        "low": "{input}와/과 포함관계에 있어요",
    },
    "시간관계": {
        "high": "{input} 시기에 일어나는 것이에요",
        "medium": "{input} 때 경험하는 것이에요",
        "low": "{input}와/과 시간적으로 연결되어 있어요",
    },
    "결과관계": {
        "high": "{input}의 결과로 나타나는 것이에요",
        "medium": "{input} 이후에 생기는 것이에요",
        "low": "{input}에서 파생된 것이에요",
    },
}

# 세부 점수 기반 추가 힌트 (기존 시스템과 병행)
DETAIL_HINT_SUFFIX = {
    "semantic_high": "의미적으로 매우 가까워요",
    "relational_high": "맥락이나 상황은 정확해요",
    "formative_high": "철자가 거의 비슷해요",
    "contradiction": "하지만 정반대 의미는 아니에요",
}