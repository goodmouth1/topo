import sys, pickle, numpy as np

pkl_file = sys.argv[1]
with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)

print("\n" + "🔍"*20)
print(f"📂 파일명: {pkl_file}")
print(f"총 담긴 데이터 덩어리 개수: {len(preds)}개")

for i in range(min(len(preds), 5)):  # 너무 길면 안 되니 5개까지만 확인
    data = preds[i]
    try:
        shape_info = np.shape(data)
    except:
        shape_info = len(data) if hasattr(data, '__len__') else 'No shape'
    
    print(f"\n[{i}번 칸] 타입: {type(data)} | 크기/모양: {shape_info}")
    
    # 앞부분 3개만 살짝 출력해서 정체 파악하기
    if hasattr(data, '__getitem__'):
        try:
            print(f" 👉 샘플: {data[:3]}")
        except:
            print(" 👉 샘플 출력 불가")
print("\n" + "🔍"*20 + "\n")
