from gazetteer import Gazetteer
from haversine import haversine

# 1. 지명 사전 로딩 (시간이 좀 걸리지만 한 번만 하면 됨)
gaz = Gazetteer()

def solve_ambiguity(user_predicted_lat, user_predicted_lon, mention_text):
    """
    [논문의 핵심 로직]
    사용자 예측 위치(Lat, Lon)와 언급된 지명(Text)을 주면,
    거리 기반으로 가장 유력한 지명을 찾아냄.
    """
    print(f"\n--------------------------------------------------")
    print(f"[상황] AI가 예측한 사용자 위치: ({user_predicted_lat}, {user_predicted_lon})")
    print(f"[입력] 사용자가 언급한 단어: '{mention_text}'")
    
    # 1. 후보군 검색 (Gazetteer)
    candidates = gaz.get_candidates(mention_text, max_results=10)
    
    if not candidates:
        print("-> 후보 지명을 찾을 수 없습니다.")
        return

    # 2. 거리 계산 (Distance Calculation)
    # 논문 가설: "사용자와 물리적으로 더 가까운 지명일 확률이 높다"
    user_loc = (user_predicted_lat, user_predicted_lon)
    
    print(f"[분석] '{mention_text}' 후보군 {len(candidates)}개 거리 계산 중...")
    
    for c in candidates:
        cand_loc = (c['lat'], c['lon'])
        # 하버사인(Haversine) 공식으로 지구상 실제 거리 계산 (km)
        dist = haversine(user_loc, cand_loc)
        c['distance_km'] = dist

    # 3. 거리순 정렬 (가장 가까운 게 0번으로 오게)
    candidates.sort(key=lambda x: x['distance_km'])
    
    # 4. 결과 출력
    winner = candidates[0]
    print(f"\n[결과] AI가 선택한 정답: ★ {winner['name']} ({winner['state']}, {winner['country']}) ★")
    print(f"       -> 사용자와의 거리: {winner['distance_km']:.2f} km")
    
    # 오답 후보들 비교
    if len(candidates) > 1:
        runner_up = candidates[1]
        print(f"       (2등 후보: {runner_up['name']}, {runner_up['state']} - {runner_up['distance_km']:.2f} km 떨어짐)")

# --- 시뮬레이션 테스트 ---
if __name__ == "__main__":
    
    # 시나리오 1: 사용자가 '미국 오리건주(Oregon)'에 살고 있다고 가정
    # 오리건주 좌표 대략: (44.0, -123.0)
    solve_ambiguity(44.0, -123.0, "Springfield")
    
    # 시나리오 2: 사용자가 '미국 매사추세츠주(Massachusetts)'에 살고 있다고 가정
    # 매사추세츠 좌표 대략: (42.1, -72.5)
    solve_ambiguity(42.1, -72.5, "Springfield")
    
    # 시나리오 3: 사용자가 '대한민국 서울'에 살고 있다고 가정
    # 서울 좌표: (37.56, 126.97)
    solve_ambiguity(37.56, 126.97, "Springfield")