import pickle
import os
import sys

# 1. 파일 경로 설정
INPUT_FILE = "test_with_ground_truth.pkl"  # 아까 만든(후보군이 포함된) 파일
OUTPUT_FILE = "manual_labeled_data.pkl"    # 호날두님이 직접 라벨링한 결과

print("✍️ [수동 라벨링 도구] 논문의 품격을 높이는 작업 시작!\n")

# 2. 데이터 로드
if not os.path.exists(INPUT_FILE):
    print(f"❌ 오류: {INPUT_FILE} 파일이 없습니다. create_ground_truth.py를 먼저 돌리세요.")
    sys.exit()

with open(INPUT_FILE, 'rb') as f:
    all_data = pickle.load(f)

# 이미 작업한 내용이 있으면 불러오기 (이어하기 기능)
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
    for i in range(start_idx, total_count):
        item = all_data[i]
        user_id = item['user_id']
        toponyms = item['toponyms']
        
        # 지명 중의성이 있는 케이스만 진행
        if not toponyms:
            continue
            
        print("\n" + "="*60)
        print(f"🚀 Progress: [{i+1}/{total_count}] (현재까지 확보된 골드 데이터: {len(labeled_data)}개)")
        print(f"👤 User ID: {user_id}")
        print(f"📍 유저 실제 집 위치(참고용): {item['user_true_loc']}")
        print("-" * 60)
        
        new_toponyms = []
        skip_user = False
        
        for topo in toponyms:
            word = topo['word']
            candidates = topo['candidates']
            
            # 후보가 1개면 라벨링할 필요 없음 (자동 통과)
            if len(candidates) < 2:
                topo['human_label_idx'] = 0
                new_toponyms.append(topo)
                continue
                
            print(f"\n🗣️  언급 단어: [ {word} ]")
            print("👇 정답을 선택하세요 (문맥상 어디일까요?):")
            
            for idx, cand in enumerate(candidates):
                print(f"   [{idx}] {word} ({cand['state']}, US) - 거리: {cand.get('dist_from_user', 'N/A')}")
            
            print("   [s] Skip (모르겠음/애매함)")
            print("   [q] 저장하고 종료")
            
            while True:
                choice = input("👉 선택 (번호 입력): ").strip().lower()
                
                if choice == 'q':
                    raise KeyboardInterrupt # 강제 종료 루틴으로 이동
                elif choice == 's':
                    print("⏩ 스킵합니다.")
                    skip_user = True
                    break
                elif choice.isdigit() and 0 <= int(choice) < len(candidates):
                    # 선택한 후보를 'Human Label'로 저장
                    selected_idx = int(choice)
                    topo['human_label_idx'] = selected_idx
                    new_toponyms.append(topo)
                    print(f"✅ 선택 완료: {candidates[selected_idx]['name']}")
                    break
                else:
                    print("❌ 잘못된 입력입니다. 다시 입력하세요.")
            
            if skip_user: break
        
        # 스킵되지 않은 유저만 저장
        if not skip_user:
            item['toponyms'] = new_toponyms # 사람의 선택 정보가 추가됨
            labeled_data.append(item)
            
        # 10명마다 자동 저장 (날아감 방지)
        if len(labeled_data) % 10 == 0:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(labeled_data, f)
            print("💾 자동 저장 완료.")

except KeyboardInterrupt:
    print("\n\n🛑 작업 중단! 현재까지 진행 상황을 저장합니다.")

# 4. 최종 저장
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(labeled_data, f)

print("\n" + "="*60)
print(f"🎉 수고하셨습니다! 총 {len(labeled_data)}개의 골드 데이터가 '{OUTPUT_FILE}'에 저장되었습니다.")
print("="*60)