import gzip
import pickle

print("1. World 압축파일 해부 중...")
with gzip.open('./data/world/world.pkl.gz', 'rb') as f:
    data = pickle.load(f, encoding='latin1')

print("\n✅ 숨겨진 변수 목록:")
if hasattr(data, '__dict__'):
    print(list(data.__dict__.keys()))
elif isinstance(data, dict):
    print(list(data.keys()))
elif isinstance(data, tuple) or isinstance(data, list):
    print(f"데이터가 튜플/리스트입니다. 길이: {len(data)}")
    for i, item in enumerate(data):
        print(f" - Index {i} 형태: {type(item)}")

print("\n2. AI 채점지 해부 중...")
with open('gcn_1.0_percent_pred_930.pkl', 'rb') as f:
    preds = pickle.load(f)
print("형태:", type(preds))
if isinstance(preds, (list, tuple)):
    print("길이:", len(preds))
    print("첫번째 요소 형태:", type(preds[0]))
    if hasattr(preds[0], 'shape'):
        print("첫번째 요소 shape:", preds[0].shape)
