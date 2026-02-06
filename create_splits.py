# create_splits.py
import os
import csv

# 경로 설정 (본인의 폴더 구조에 맞게 수정하세요)
BASE_DIR = '/Users/coys/Library/CloudStorage/GoogleDrive-wangfubu@gmail.com/내 드라이브/realu/master_thesis/data/cmu'
FULL_TEXT_PATH = os.path.join(BASE_DIR, 'full_text.txt')

# 출력 파일 경로
TRAIN_FILE = os.path.join(BASE_DIR, 'user_info.train')
DEV_FILE = os.path.join(BASE_DIR, 'user_info.dev')
TEST_FILE = os.path.join(BASE_DIR, 'user_info.test')

def generate_splits():
    print(f"Reading from {FULL_TEXT_PATH}...")
    
    # 사용자별 위치 정보를 저장할 딕셔너리
    # user_data = { 'USER_ID': (lat, lon) }
    user_data = {}
    
    try:
        with open(FULL_TEXT_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 5:
                    continue
                
                u_id = parts[0]
                # full_text.txt의 3번째, 4번째 컬럼이 위도, 경도입니다.
                # (0-index 기준: 3=lat, 4=lon)
                try:
                    lat = parts[3]
                    lon = parts[4]
                    # 한 명의 유저가 여러 트윗을 썼지만, 집 위치는 동일하므로 첫 번째 것만 저장하거나 덮어씁니다.
                    user_data[u_id] = (lat, lon)
                except IndexError:
                    continue
                    
    except FileNotFoundError:
        print(f"[Error] {FULL_TEXT_PATH} 파일을 찾을 수 없습니다.")
        print("data/cmu/ 폴더 안에 full_text.txt 파일이 있는지 확인해주세요.")
        return

    print(f"Total unique users found: {len(user_data)}")

    # 파일 핸들 열기
    f_train = open(TRAIN_FILE, 'w', encoding='utf-8')
    f_dev = open(DEV_FILE, 'w', encoding='utf-8')
    f_test = open(TEST_FILE, 'w', encoding='utf-8')
    
    counts = {'train': 0, 'dev': 0, 'test': 0}

    # README.txt 규칙에 따라 분할
    # "Train is folds 1,2,3 / Dev is fold 4 / Test is fold 5"
    # fold = (userID % 5). 
    # 나머지: 1, 2, 3 -> Train
    # 나머지: 4 -> Dev
    # 나머지: 0 -> Test (5와 동일)
    
    for u_id, (lat, lon) in user_data.items():
        # 유저 ID 형식: "USER_79321756" -> 숫자 부분만 추출 (16진수)
        try:
            hex_id = u_id.split('_')[1]
            numeric_id = int(hex_id, 16) # 16진수를 정수로 변환
            
            remainder = numeric_id % 5
            
            line = f"{u_id}\t{lat}\t{lon}\n"
            
            if remainder in [1, 2, 3]:
                f_train.write(line)
                counts['train'] += 1
            elif remainder == 4:
                f_dev.write(line)
                counts['dev'] += 1
            else: # remainder == 0
                f_test.write(line)
                counts['test'] += 1
                
        except Exception as e:
            print(f"Skipping user {u_id}: {e}")

    # 파일 닫기
    f_train.close()
    f_dev.close()
    f_test.close()

    print("-" * 30)
    print(f"Splits generated successfully!")
    print(f"  - Train: {counts['train']} users -> {TRAIN_FILE}")
    print(f"  - Dev:   {counts['dev']} users -> {DEV_FILE}")
    print(f"  - Test:  {counts['test']} users -> {TEST_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    generate_splits()