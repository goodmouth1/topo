import sys, pickle, numpy as np

pkl_file = sys.argv[1]
with open(pkl_file, 'rb') as f:
    preds = pickle.load(f)

# preds[0]에는 이미 완벽하게 계산된 거리(km) 배열이 들어있습니다!
distances = np.array(preds[0])

# 거리들을 가지고 161km 적중률과 평균/중간 오차를 바로 구합니다.
acc_161 = np.mean(distances <= 161.0) * 100
mean_err = np.mean(distances)
median_err = np.median(distances)

print("\n" + "🔥"*25)
print(f"🏆 진짜 팩트 성적표: {pkl_file} 🏆")
print(f"🎯 거리 정확도 (Acc @ 161km): {acc_161:.2f} %")
print(f"📐 평균 오차 (Mean Error)   : {mean_err:.2f} km")
print(f"📏 중간 오차 (Median Error) : {median_err:.2f} km")
print("🔥"*25 + "\n")
