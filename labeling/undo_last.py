import pickle
import os

# 파일 경로
FILE_NAME = "manual_labeled_data.pkl"

if not os.path.exists(FILE_NAME):
    print("❌ 파일이 없습니다. 지울 데이터가 없어요.")
else:
    # 1. 파일 불러오기
    with open(FILE_NAME, 'rb') as f:
        data = pickle.load(f)
    
    if len(data) > 0:
        # 2. 마지막 데이터 꺼내서 확인하기
        last_item = data.pop() # 리스트의 맨 마지막 요소를 뽑아내고 삭제함
        user_id = last_item['user_id']
        
        print(f"🗑️ [삭제 완료] 마지막으로 작업한 유저를 리스트에서 뺐습니다.")
        print(f"   - 삭제된 유저 ID: {user_id}")
        
        # 3. 다시 저장하기
        with open(FILE_NAME, 'wb') as f:
            pickle.dump(data, f)
            
        print(f"✅ 이제 라벨링 도구를 다시 실행하면 '{user_id}'부터 다시 시작합니다!")
        print(f"   (현재 저장된 데이터 개수: {len(data)}개)")
    else:
        print("⚠️ 데이터가 비어있어서 지울 게 없습니다.")