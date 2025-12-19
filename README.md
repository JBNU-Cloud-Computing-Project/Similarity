# WordTreasure 유사도 계산 서비스

AI 기반 한국어 단어 유사도 측정 및 **맥락적 힌트 생성** 서비스

## 📋 Phase 1+2 Enhanced 구현 기능

### ✅ 구현 완료
- **의미 유사도 (Semantic)**: 문장 임베딩 기반 코사인 유사도
- **관계 유사도 (Relational)**: NLI 모델 기반 맥락 관계 분석
- **형태 유사도 (Formative)**: 자모 분해 편집거리 계산
- **🌟 맥락적 힌트 (NEW)**: 입력-정답 간 관계 분석 기반 자연스러운 힌트
- **🌟 관계 유형 분석 (NEW)**: 10가지 관계 패턴 자동 인식
- **반의어 감점**: 모순 관계 자동 감지

### 💡 맥락적 힌트 시스템

**기존 방식:**
```
입력: "친구" → 정답: "배신"
힌트: "비슷한 방향이에요" ❌ (너무 일반적)
```

**새로운 방식:**
```
입력: "친구" → 정답: "배신"
관계 분석: "사람관계" (확신도 0.82)
힌트: "친구 사이에서 나타나는 것이에요" ✅ (맥락적)
```

**지원하는 관계 유형:**
1. 상황발생 - "X 상황에서 나타나는 것"
2. 감정원인 - "X에서 비롯되는 감정"
3. 속성관계 - "X의 성질을 가진 것"
4. 사람관계 - "X 사이에서 발생하는 것"
5. 유사장르 - "X와 비슷한 방식"
6. 반대관계 - "X와 반대되는 것"
7. 장소관계 - "X에서 경험할 수 있는 것"
8. 시간관계 - "X 시기에 일어나는 것"
9. 부분전체 - "X의 한 부분"
10. 결과관계 - "X의 결과로 나타나는 것"

### 🔜 Phase 3 예정
- 도메인 게이팅 (게임/음식/감정 등)
- 카테고리 특화 메카닉 분석
- 캘리브레이션 시스템

---

## 🚀 빠른 시작

### 1. 환경 설정

**Python 3.9 이상 필요**

```bash
# 가상환경 생성 (권장)
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

### 2. 서버 실행

```bash
python main.py
```

**첫 실행 시:**
- 모델 자동 다운로드 (~1.5GB)
- 약 30-60초 소요
- 이후 실행은 즉시 시작

**실행 확인:**
```
🚀 WordTreasure 유사도 서비스 시작
📦 모델 로딩 시작... (약 30초 소요)
✅ Semantic 모델 로딩 완료
✅ NLI 모델 로딩 완료
🚀 모든 모델 로딩 완료!
🌐 서버 준비 완료: http://0.0.0.0:8000
```

---

## 🧪 테스트

### curl로 테스트

```bash
# 예시 1: 친구 → 배신
curl -X POST "http://localhost:8000/api/similarity/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "친구",
    "answer": "배신"
  }'

# 예시 2: 마피아 → 라이어 게임
curl -X POST "http://localhost:8000/api/similarity/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "마피아",
    "answer": "라이어 게임"
  }'

# 예시 3: 정답 입력
curl -X POST "http://localhost:8000/api/similarity/calculate" \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "배신",
    "answer": "배신"
  }'
```

### 응답 예시

**예시 1: 맥락적 힌트 (사람관계)**
```bash
curl -X POST "http://localhost:8000/api/similarity/calculate" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "친구", "answer": "배신"}'
```
```json
{
  "similarity_score": 45.32,
  "hint": "친구 사이에서 나타나는 것이에요",
  "category_match": false,
  "breakdown": {
    "semantic": 0.62,
    "relational": 0.38,
    "formative": 0.15,
    "contradiction": 0.05
  },
  "processing_time_ms": 145.67
}
```

**예시 2: 맥락적 힌트 (유사장르)**
```bash
curl -X POST "http://localhost:8000/api/similarity/calculate" \
  -H "Content-Type: application/json" \
  -d '{"user_input": "마피아", "answer": "라이어 게임"}'
```
```json
{
  "similarity_score": 78.45,
  "hint": "마피아와 비슷한 방식으로 진행되는 것이에요",
  "category_match": false,
  "breakdown": {
    "semantic": 0.72,
    "relational": 0.81,
    "formative": 0.35,
    "contradiction": 0.0
  },
  "processing_time_ms": 138.92
}
```

---

## 🔌 Spring Boot 연동

### 1. DTO 클래스 생성

```java
// SimilarityRequest.java
@Data
public class SimilarityRequest {
    private String userInput;
    private String answer;
}

// SimilarityResponse.java
@Data
public class SimilarityResponse {
    private Double similarityScore;
    private String hint;
    private Boolean categoryMatch;
    private Map<String, Double> breakdown;
    private Double processingTimeMs;
}
```

### 2. 서비스 구현

```java
@Service
public class PythonSimilarityCalculator implements SimilarityCalculator {
    
    private final RestTemplate restTemplate;
    private final String pythonServiceUrl = "http://localhost:8000";
    
    @Override
    public BigDecimal calculateSimilarity(String userInput, String answer) {
        SimilarityRequest request = new SimilarityRequest();
        request.setUserInput(userInput);
        request.setAnswer(answer);
        
        SimilarityResponse response = restTemplate.postForObject(
            pythonServiceUrl + "/api/similarity/calculate",
            request,
            SimilarityResponse.class
        );
        
        return BigDecimal.valueOf(response.getSimilarityScore());
    }
    
    @Override
    public String generateHint(String userInput, String answer, BigDecimal similarity) {
        // Python 서비스에서 이미 힌트를 생성하므로 재사용
        SimilarityRequest request = new SimilarityRequest();
        request.setUserInput(userInput);
        request.setAnswer(answer);
        
        SimilarityResponse response = restTemplate.postForObject(
            pythonServiceUrl + "/api/similarity/calculate",
            request,
            SimilarityResponse.class
        );
        
        return response.getHint();
    }
}
```

### 3. RestTemplate 설정

```java
@Configuration
public class RestTemplateConfig {
    
    @Bean
    public RestTemplate restTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(5000);  // 5초
        factory.setReadTimeout(10000);    // 10초
        
        return new RestTemplate(factory);
    }
}
```

### 4. 기존 TemporarySimilarityCalculator 대체

```java
// application.yml
spring:
  profiles:
    active: prod  # 또는 local

---
# local 프로필에서는 임시 계산기 사용
spring:
  config:
    activate:
      on-profile: local

@Profile("local")
@Component
class TemporarySimilarityCalculator implements SimilarityCalculator { ... }

---
# prod 프로필에서는 Python 서비스 사용
spring:
  config:
    activate:
      on-profile: prod

@Profile("prod")
@Primary
@Component  
class PythonSimilarityCalculator implements SimilarityCalculator { ... }
```

---

## 📊 API 문서

서버 실행 후 자동 생성:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

---

## ⚙️ 설정 변경

`config.py` 파일에서 조정 가능:

```python
# 가중치 조정
WEIGHTS = {
    "semantic": 0.50,      # 의미 유사도
    "relational": 0.35,    # 관계 유사도
    "formative": 0.15,     # 형태 유사도
}

# NLI 템플릿 추가/수정
NLI_TEMPLATES = [
    "{input}은 {answer}과 관련이 있다.",
    # 새 템플릿 추가...
]
```

---

## 🐛 문제 해결

### 모델 다운로드 실패
```bash
# Hugging Face 토큰 설정 (필요시)
export HF_TOKEN="your_token_here"
```

### 메모리 부족
```python
# config.py에서 더 작은 모델로 변경
SEMANTIC_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
```

### 응답 시간 느림
- GPU 사용 권장 (config.py에서 device=0)
- 또는 배치 처리 구현

---

## 📈 성능 목표

- **응답 시간**: < 150ms (p95)
- **첫 요청**: < 200ms (모델 캐시 후)
- **메모리 사용**: ~2GB

---

## 🔄 다음 단계 (Phase 3)

1. 도메인 게이팅 시스템
2. Redis 캐싱
3. 배치 처리
4. 로그 수집 및 분석
5. AWS 배포 (Lambda/EC2)

---

## 📝 라이선스

MIT License