import data
import pickle
import numpy as np
from sklearn.metrics import accuracy_score
import os

print("Loading data...")
# 1. 데이터 로드
dataset = data.DataLoader(data_home='./data/', bucket_size=50)
dataset.load_data()

# -----------------------------------------------------------
# [자동 탐지 로직] 변수 이름이 달라도 알아서 찾습니다.
# -----------------------------------------------------------
test_idx = None
test_label = None

# 1. Test Index 찾기 (후보군: test_idx, idx_test, test_ids)
possible_idx_names = ['test_idx', 'idx_test', 'test_ids', 'ids_test', 'test_user_idx']
for name in possible_idx_names:
    if hasattr(dataset, name):
        test_idx = getattr(dataset, name)
        print(f"✅ Found Test Index variable: '{name}'")
        break

# 2. Test Label 찾기 (후보군: test_label, labels_test, y_test, test_y)
possible_label_names = ['test_label', 'labels_test', 'y_test', 'test_y', 'label_test']
for name in possible_label_names:
    if hasattr(dataset, name):
        test_label = getattr(dataset, name)
        print(f"✅ Found Test Label variable: '{name}'")
        break

# 3. 그래도 못 찾았으면 목록을 보여주고 종료
if test_idx is None or test_label is None:
    print("\n" + "="*50)
    print("❌ 변수 이름을 자동으로 찾지 못했습니다.")
    print("dataset 객체 안에 있는 변수 목록은 아래와 같습니다.")
    print("이 중에서 'Test Index'와 'Test Label'로 보이는 이름을 찾아주세요.")
    print("="*50)
    # dataset 안에 있는 모든 변수 이름 출력
    print(list(dataset.__dict__.keys()))
    print("="*50 + "\n")
    exit()
# -----------------------------------------------------------

print(f"Test Data Size: {len(test_idx)}")

print("Loading best model...")
model_path = 'model.pkl'

if not os.path.exists(model_path):
    print(f"Error: {model_path} 파일을 찾을 수 없습니다.")
else:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    print("Predicting...")
    # 3. 수능 치기
    preds = model.predict(test_idx)

    # 4. 채점
    acc = accuracy_score(test_label, preds)
    print("\n" + "="*40)
    print(f"🏆 FINAL TEST ACCURACY: {acc * 100:.2f}%")
    print("="*40 + "\n")