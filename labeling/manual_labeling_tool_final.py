import pickle
import os
import sys

# 1. 파일 자동 탐색 (LLM 버전 우선, 없으면 일반 버전)
FILES_TO_CHECK = ["test_with_ground_truth_llm.pkl", "test_with_ground_truth.pkl"]
INPUT_FILE = None

for fname in FILES_TO_CHECK:
    if os.path.exists(fname):
        INPUT_FILE = fname
        break

if INPUT_FILE is None:
    print(f"❌ 오류: 정답지 파일이 없습니다. create_ground_truth 스크립트를 먼저 실행했는지 확인하세요.")
    sys.exit()

OUTPUT_FILE = "manual_labeled_data.pkl"

# 색상 코드
YELLOW = '\033[93m'
BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[92m'
RED = '\033[91m'

print(f"{BOLD}✍️ [수동 라벨링 도구] 파일 로드됨: {INPUT_FILE}{END}")
print(f"   (총 556개 중 '진짜'만 골라내는 작업입니다!)\n")

with open(INPUT_FILE, 'rb') as f:
    all_data = pickle.load(f)

# 이어하기 기능
labeled_data = []
start_idx = 0

if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'rb') as f:
        labeled_data = pickle.load(f)
        start_idx = len(labeled_data)
        print(f"📂 이전 작업({start_idx}개)을 불러왔습니다. 이어서 진행합니다...")

# 3. 라벨링 루프
total_count = len(all_data)

try:
    # 이미 처리한 데이터 수만큼 건너뛰기 (단순 인덱스 기반)
    # 주의: all_data 순서가 바뀌지 않았다고 가정
    current_processed_count = 0
    
    for i in range(total_count):
        # 이미 라벨링한 개수만큼은 스킵 (대략적인 이어하기)
        if current_processed_count < start_idx:
            current_processed_count += 1
            continue
            
        item = all_data[i]
        user_id = item['user_id']
        toponyms = item['toponyms']
        
        if not toponyms: continue
            
        print("\n" + "="*80)
        print(f"🚀 Progress: [{i+1}/{total_count}] (확보된 골드 데이터: {len(labeled_data)}개)")
        print(f"👤 User ID: {user_id}")
        
        # 트윗 원문
        tweet_text = toponyms[0].get('orig_text', "텍스트 없음")
        
        print("-" * 80)
        print(f"{BOLD}[ 트윗 원문 ]{END}")
        
        display_text = tweet_text
        for topo in toponyms:
            word = topo['word']
            display_text = display_text.replace(word, f"{YELLOW}{word}{END}")
        print(f"  📢 {display_text}")
        print("-" * 80)
        
        new_toponyms = []
        skip_user = False
        
        for topo in toponyms:
            word = topo['word']
            candidates = topo['candidates']
            
            # 후보가 1개면 라벨링 불필요
            if len(candidates) < 2:
                topo['human_label_idx'] = 0
                new_toponyms.append(topo)
                continue
            
            print(f"\n🎯 타겟 단어: {YELLOW}{BOLD}[ {word} ]{END}")
            print(f"👇 이 '{word}'가 문맥상 어디인지 선택하세요:")
            
            for idx, cand in enumerate(candidates):
                # 거리 정보가 있으면 표시
                dist_str = ""
                if 'dist_from_user' in cand:
                    dist_str = f"- 유저와 {cand['dist_from_user']:.1f}km"
                    
                star = "⭐" if idx == 0 else "  " # 0번이 보통 제일 가까운 곳(추천)
                print(f"   [{idx}] {star} {BOLD}{cand['name']}{END} ({cand['state']}, US) {dist_str}")
            
            print(f"   [{RED}s{END}] Skip (지명이 아님 / 사람 이름 / 모르겠음)")
            print(f"   [q] 저장하고 종료")
            
            valid_input = False
            while not valid_input:
                choice = input("👉 선택: ").strip().lower()
                
                if choice == 'q':
                    raise KeyboardInterrupt
                elif choice == 's':
                    print(f"{RED}⏩ 제외합니다.{END}")
                    valid_input = True
                elif choice.isdigit() and 0 <= int(choice) < len(candidates):
                    selected_idx = int(choice)
                    topo['human_label_idx'] = selected_idx
                    new_toponyms.append(topo)
                    print(f"{GREEN}✅ 선택 완료!{END}")
                    valid_input = True
                else:
                    print("❌ 다시 입력해주세요.")
        
        # 하나라도 건진 게 있으면 저장
        if new_toponyms:
            item['toponyms'] = new_toponyms
            labeled_data.append(item)
            current_processed_count += 1 # 처리 카운트 증가
            
        # 자동 저장
        if len(labeled_data) % 5 == 0 and len(labeled_data) > 0:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(labeled_data, f)
            print("💾 (자동 저장됨)")

except KeyboardInterrupt:
    print("\n\n🛑 작업 중단! 저장 중...")

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(labeled_data, f)

print(f"\n🎉 총 {len(labeled_data)}개의 검증된 골드 데이터 확보!")
print(f"📁 저장 파일: {OUTPUT_FILE}")
