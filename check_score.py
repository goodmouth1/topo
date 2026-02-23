import sys
import pickle
import numpy as np

# 1. 터미널에서 넘겨준 파일 이름(gcn_1.0_percent_pred_930.pkl)을 정확히 받아옵니다.
pkl_file = sys.argv[1]
print(f"Loading predictions from {pkl_file}...")

with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)

# 2. AI 예측값 (preds[0])
y_p = np.array(preds[0].todense()) if hasattr(preds[0], 'todense') else np.array(preds[0])
y_pred = np.argmax(y_p, axis=1) if len(np.shape(y_p)) > 1 else y_p.flatten()

# 3. 진짜 정답값 (preds[1])
y_t = np.array(preds[1].todense()) if hasattr(preds[1], 'todense') else np.array(preds[1])
y_true = np.argmax(y_t, axis=1) if len(np.shape(y_t)) > 1 else y_t.flatten()

# 4. 심플하고 완벽한 채점
acc_class = np.mean(y_pred == y_true) * 100

print("\n" + "="*50)
print(f"🏆 GCN 모델 최종 정확도 (Class Accuracy) 🏆")
print(f"✅ 구역 적중률: {acc_class:.2f}%")
print("="*50 + "\n")