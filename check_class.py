import sys
import pickle
import numpy as np
import gzip

pkl_file = sys.argv[1]

# 1. AI가 푼 시험지 로드
with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)
    
y_pred = preds[0] if isinstance(preds, tuple) else preds
if len(np.shape(y_pred)) > 1:
    y_pred = np.argmax(y_pred, axis=1)

# 2. World 원본 정답지 로드
with gzip.open('./data/world/world.pkl.gz', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# 3. 넘파이 배열(정답지)만 쏙 뽑아서 길이 맞는 것 찾기
arrays = [x for x in data if isinstance(x, np.ndarray)]
try:
    y_test = [a for a in arrays if len(a) == len(y_pred)][0]
except IndexError:
    print("❌ 에러: 시험지와 정답지의 문제 수가 다릅니다!")
    sys.exit()

if len(np.shape(y_test)) > 1:
    y_test = np.argmax(y_test, axis=1)

# 4. 심플하고 확실한 채점!
acc = np.mean(y_pred == y_test) * 100

print("\n" + "🔥"*25)
print(f"🌍 World 모델 구역 적중률 (Classification) 🌍")
print(f"🎯 최종 Accuracy : {acc:.2f} %")
print("🔥"*25 + "\n")