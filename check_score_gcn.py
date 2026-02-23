import pickle
import numpy as np
import sys
import os

def check_score(filename):
    print(f"\n📂 파일 분석 중: {filename} ...")
    
    if not os.path.exists(filename):
        print("❌ 파일을 찾을 수 없습니다.")
        return

    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            
        # 데이터 구조 확인 (튜플 형태 예상)
        # data[0]: 거리 오차(Error distances)
        # data[1]: 예측 좌표(Predicted coordinates) - 안 쓰더라도 구조 확인용
        
        if isinstance(data, (tuple, list)) and len(data) >= 1:
            errors = np.array(data[0])
        else:
            # 구조가 다를 경우 바로 에러 처리라고 가정
            errors = np.array(data)

        # ---------------------------------------------------------
        # 📝 논문용 핵심 지표 계산
        # ---------------------------------------------------------
        total_samples = len(errors)
        acc_161 = np.mean(errors <= 161.0) * 100  # Accuracy @ 161km
        mean_error = np.mean(errors)              # Mean Error
        median_error = np.median(errors)          # Median Error

        print("-" * 40)
        print(f"✅ 분석 결과 (Total Samples: {total_samples})")
        print("-" * 40)
        print(f"🎯 Accuracy @ 161km : {acc_161:.2f}%  <-- (논문용)")
        print(f"📏 Mean Error       : {mean_error:.2f} km")
        print(f"wv Median Error     : {median_error:.2f} km")
        print("-" * 40)

    except Exception as e:
        print(f"❌ 에러 발생: {e}")
        print("파일이 손상되었거나 아직 생성 중일 수 있습니다.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # 파일 이름 안 넣었으면 자동으로 가장 최신 pkl 파일 찾기
        pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl') and 'pred' in f]
        if pkl_files:
            latest_file = max(pkl_files, key=os.path.getmtime)
            check_score(latest_file)
        else:
            print("사용법: python check_score_gcn.py [파일이름.pkl]")
    else:
        check_score(sys.argv[1])