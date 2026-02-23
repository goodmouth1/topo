import sys, pickle, numpy as np

with open(sys.argv[1], 'rb') as f:
    preds = pickle.load(f)

y_p = np.array(preds[0].todense()) if hasattr(preds[0], 'todense') else np.array(preds[0])
y_pred = np.argmax(y_p, axis=1) if len(np.shape(y_p)) > 1 else y_p.flatten()

print("\n" + "🔍"*20)
print(f"📊 테스트 한 총 유저 수 : {len(y_pred)}명")
print(f"🤖 AI가 찍은 구역 번호들 : {np.unique(y_pred)}")
print(f"⚠️ 확률 값에 NaN(결측치)이 터졌는가? : {np.isnan(y_p).any()}")
print("🔍"*20 + "\n")
