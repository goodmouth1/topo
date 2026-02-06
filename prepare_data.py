# prepare_data.py
import pandas as pd
import os
import gzip

# 경로 설정
BASE_DIR = '/Users/coys/Library/CloudStorage/GoogleDrive-wangfubu@gmail.com/내 드라이브/realu/master_thesis/data/cmu'
FULL_TEXT = os.path.join(BASE_DIR, 'full_text.txt')

def merge_and_save(split_name):
    user_file = os.path.join(BASE_DIR, f'user_info.{split_name}')
    output_file = os.path.join(BASE_DIR, f'user_info.{split_name}.gz')

    print(f"Processing {split_name} data...")

    # 1. 유저 ID 및 좌표 로드
    try:
        users = pd.read_csv(user_file, sep='\t', header=None, names=['user', 'lat', 'lon'], usecols=[0, 1, 2])
        print(f"  - Loaded {len(users)} users from {user_file}")
    except Exception as e:
        print(f"  - [Error] {user_file} 파일을 찾을 수 없습니다: {e}")
        return

    # 2. 전체 텍스트 로드 (메모리 효율을 위해 딕셔너리 활용)
    texts = {}
    print("  - Reading full_text.txt (this may take a moment)...")
    try:
        with open(FULL_TEXT, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 6:
                    u_id = parts[0]
                    text_content = parts[-1]
                    
                    # 같은 유저의 트윗들을 이어 붙임
                    if u_id in texts:
                        texts[u_id] += " " + text_content
                    else:
                        texts[u_id] = text_content
    except FileNotFoundError:
        print(f"  - [Error] {FULL_TEXT} 파일을 찾을 수 없습니다.")
        return

    # 3. 병합 (유저 ID 기준)
    users['text'] = users['user'].map(texts).fillna("")
    
    # 텍스트가 없는 유저 제거
    before_count = len(users)
    users = users[users['text'] != ""]
    print(f"  - Removed {before_count - len(users)} users with no text.")

    # 4. 압축 저장 (.gz)
    # data.py는 탭(\t)으로 구분된 4개 컬럼(user, lat, lon, text)을 기대함
    users.to_csv(output_file, sep='\t', index=False, header=False, compression='gzip')
    print(f"  - Saved to {output_file}")

if __name__ == "__main__":
    if not os.path.exists(FULL_TEXT):
        print(f"Error: {FULL_TEXT} not found. Check your folder structure.")
    else:
        merge_and_save('train')
        merge_and_save('dev')
        merge_and_save('test')
        print("\n[성공] 모든 데이터 준비 완료! 이제 gcnmain.py를 실행하세요.")