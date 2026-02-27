import pickle
import re

print("🌍 [Phase 1] 트위터 데이터 로딩 및 파싱 시작...")

# GeoCorpora 벤치마크 스타일의 트윗 데이터 (텍스트 + 정답 좌표)
raw_twitter_data = [
    {
        "user_id": "user_geo_01",
        "text": "Heading to Dallas for the weekend! @friend_A @travel_bot",
        "toponym": "Dallas",
        "true_lat": 32.7767, "true_lon": -96.7970, "true_state": "TX"
    },
    {
        "user_id": "user_geo_02",
        "text": "The weather in Oakland is terrible today. @friend_B missing you CA!",
        "toponym": "Oakland",
        "true_lat": 37.8044, "true_lon": -122.2712, "true_state": "CA"
    },
    {
        "user_id": "user_geo_03",
        "text": "Just arrived in Springfield! @user_geo_01 let's meet up.",
        "toponym": "Springfield",
        "true_lat": 39.7817, "true_lon": -89.6501, "true_state": "IL"
    }
]

print("\n🕸️ [Phase 2] '@멘션' 추출을 통한 미니 소셜 그래프(Graph) 생성 중...")
edges = []
nodes = set()

for data in raw_twitter_data:
    user = data["user_id"]
    nodes.add(user)
    
    # 정규표현식으로 @아이디 추출 (GCN의 핵심인 '연결망' 창조)
    mentions = re.findall(r'@\w+', data["text"])
    
    for m in mentions:
        mentioned_user = m.replace("@", "")
        nodes.add(mentioned_user)
        edges.append((user, mentioned_user))
        print(f"   🔗 Edge 연결 발견: {user} ---> {mentioned_user}")

print(f"   📊 총 {len(nodes)}명의 유저 노드와 {len(edges)}개의 연결(Edge)로 이루어진 미니 그래프 완성!")

print("\n📦 [Phase 3] GCN 채점기 호환용 PKL 데이터로 변환 중...")
geocorpora_ready_data = []

for data in raw_twitter_data:
    # 아까 우리가 썼던 manual_labeled_data.pkl 과 완벽하게 동일한 구조로 조립!
    formatted_item = {
        'user_id': data['user_id'],
        'orig_text': data['text'],
        'user_true_loc': [data['true_lat'], data['true_lon']],
        'toponyms': [
            {
                'word': data['toponym'],
                'human_label_idx': 0, # 샘플이므로 무조건 0번이 정답이라고 가정
                'candidates': [
                    {'name': data['toponym'], 'state': data['true_state'], 'lat': data['true_lat'], 'lon': data['true_lon']},
                    {'name': data['toponym'], 'state': 'XX', 'lat': 0.0, 'lon': 0.0} # 가짜 오답 후보
                ]
            }
        ],
        'extracted_mentions': re.findall(r'@\w+', data["text"]) # 나중에 GCN에 넣을 용도
    }
    geocorpora_ready_data.append(formatted_item)

# PKL 파일로 예쁘게 저장
output_filename = 'geocorpora_sample.pkl'
with open(output_filename, 'wb') as f:
    pickle.dump(geocorpora_ready_data, f)

print(f"\n✅ 완료! GCN에 바로 밀어넣을 수 있는 '{output_filename}' 파일이 생성되었습니다.")
