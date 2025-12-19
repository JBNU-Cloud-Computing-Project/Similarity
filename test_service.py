"""
유사도 계산 서비스 테스트 스크립트
"""

import requests
import json
import time

# 서버 URL
BASE_URL = "http://localhost:8000"

def test_health():
    """헬스 체크 테스트"""
    print("\n" + "="*50)
    print("🏥 헬스 체크")
    print("="*50)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")

def test_similarity(user_input: str, answer: str):
    """유사도 계산 테스트"""
    print("\n" + "="*50)
    print(f"📝 테스트: '{user_input}' → '{answer}'")
    print("="*50)
    
    start_time = time.time()
    
    response = requests.post(
        f"{BASE_URL}/api/similarity/calculate",
        json={
            "user_input": user_input,
            "answer": answer
        }
    )
    
    elapsed = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ 성공!")
        print(f"유사도 점수: {result['similarity_score']}%")
        print(f"힌트: {result['hint']}")
        print(f"세부 점수:")
        for key, value in result['breakdown'].items():
            print(f"  - {key}: {value:.4f}")
        print(f"처리 시간: {elapsed:.2f}ms (실제 서버: {result['processing_time_ms']:.2f}ms)")
    else:
        print(f"❌ 실패: {response.status_code}")
        print(response.text)

def run_all_tests():
    """모든 테스트 실행"""
    print("\n🚀 WordTreasure 유사도 서비스 테스트 시작\n")
    
    # 헬스 체크
    try:
        test_health()
    except Exception as e:
        print(f"❌ 헬스 체크 실패: {e}")
        print("💡 서버가 실행 중인지 확인하세요: python main.py")
        return
    
    # 테스트 케이스들
    test_cases = [
        # (사용자 입력, 정답, 예상 범위)
        ("친구", "배신", "40-60% - 사람관계 맥락적 힌트"),
        ("마피아", "라이어 게임", "70-85% - 유사장르 맥락적 힌트"),
        ("배신", "배신", "100% - 정답"),
        ("게임", "라이어 게임", "30-50% - 속성관계 힌트"),
        ("거짓말", "배신", "60-75% - 감정원인 힌트"),
        ("행복", "불행", "20-40% - 반대관계 힌트"),
        ("사과", "사괴", "85-95% - 오타(형태) 힌트"),
        ("슬픔", "우울", "65-80% - 감정원인 힌트"),
    ]
    
    print("\n" + "="*50)
    print("📊 테스트 케이스 실행")
    print("="*50)
    
    results = []
    total_time = 0
    
    for user_input, answer, expected in test_cases:
        try:
            start = time.time()
            test_similarity(user_input, answer)
            elapsed = (time.time() - start) * 1000
            total_time += elapsed
            results.append((user_input, answer, "✅", elapsed))
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            results.append((user_input, answer, "❌", 0))
        
        time.sleep(0.5)  # 서버 부하 방지
    
    # 결과 요약
    print("\n" + "="*50)
    print("📈 테스트 결과 요약")
    print("="*50)
    
    success_count = sum(1 for r in results if r[2] == "✅")
    print(f"총 테스트: {len(results)}개")
    print(f"성공: {success_count}개")
    print(f"실패: {len(results) - success_count}개")
    
    if success_count > 0:
        avg_time = total_time / success_count
        print(f"평균 응답 시간: {avg_time:.2f}ms")
        
        if avg_time < 150:
            print("✅ 목표 응답 시간 달성! (< 150ms)")
        else:
            print(f"⚠️ 목표 응답 시간 초과 (목표: 150ms, 실제: {avg_time:.2f}ms)")

if __name__ == "__main__":
    run_all_tests()