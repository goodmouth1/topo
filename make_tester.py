import os

# 1. 원본 코드 읽기
with open('gcnmain.py', 'r') as f:
    lines = f.readlines()

new_lines = []

# 2. 코드 수정하기
for line in lines:
    if '.fit(' in line:
        new_lines.append(f"# [SKIPPED by Test Mode] {line}")
        
        indent = line[:line.find(line.lstrip())]
        
        # [핵심 수정] X와 H(A)를 찾을 때 여러 이름을 시도함 + 디버깅 출력 추가
        code_block = f"""
{indent}# --- Test Mode Block Start ---
{indent}import pickle
{indent}import gzip
{indent}import lasagne
{indent}from sklearn.metrics import accuracy_score

{indent}print("\\n[Test Mode] Loading best weights (Gzip)...")
{indent}with gzip.open('model.pkl', 'rb') as f:
{indent}    saved_weights = pickle.load(f)

{indent}print("[Test Mode] Injecting weights into 'l_out'...")
{indent}if 'clf' in locals():
{indent}    model_instance = locals()['clf']
{indent}elif 'model' in locals():
{indent}    model_instance = locals()['model']
{indent}else:
{indent}    print("Error: 모델 객체(clf)를 찾을 수 없습니다.")
{indent}    exit()

{indent}# [수술 진행] l_out에 가중치 주입
{indent}try:
{indent}    if hasattr(model_instance, 'l_out'):
{indent}        lasagne.layers.set_all_param_values(model_instance.l_out, saved_weights)
{indent}        print("✅ Weights injected successfully into 'l_out'!")
{indent}    else:
{indent}        print("❌ Error: 'l_out' 속성이 없습니다.")
{indent}        exit()
{indent}except Exception as e:
{indent}    print(f"Error injecting weights: {{e}}")
{indent}    exit()

{indent}print("[Test Mode] Preparing Test Data...")
{indent}# 1. Test Index 찾기
{indent}target_idx = locals().get('test_indices')
{indent}if target_idx is None:
{indent}    for name in ['test_idx', 'idx_test', 'test_ids', 'ids_test']:
{indent}        if name in locals():
{indent}            target_idx = locals()[name]
{indent}            break

{indent}# 2. 정답(Label) 추출
{indent}if 'Y' in locals():
{indent}    full_Y = locals()['Y']
{indent}    target_label = full_Y[target_idx]
{indent}    if len(target_label.shape) > 1:
{indent}        target_label = np.argmax(target_label, axis=1)
{indent}else:
{indent}    print("⚠️ Warning: 정답 변수 'Y'를 못 찾았습니다.")
{indent}    target_label = None

{indent}# 3. [핵심] X와 A(Graph) 찾기 (여러 이름 시도)
{indent}input_X = None
{indent}for name in ['X', 'features', 'x_data', 'inputs']:
{indent}    if name in locals():
{indent}        input_X = locals()[name]
{indent}        print(f"✅ Found Feature Variable: '{{name}}'")
{indent}        break

{indent}input_A = None
{indent}for name in ['H', 'A', 'adj', 'graph', 'adjacency', 'A_mat']:
{indent}    if name in locals():
{indent}        input_A = locals()[name]
{indent}        print(f"✅ Found Graph Variable: '{{name}}'")
{indent}        break

{indent}if input_X is None or input_A is None:
{indent}    print("\\n" + "!"*50)
{indent}    print("❌ Error: 예측에 필요한 변수를 못 찾았습니다.")
{indent}    print(f"   - X Found? {{input_X is not None}}")
{indent}    print(f"   - A Found? {{input_A is not None}}")
{indent}    print("🔍 현재 사용 가능한 변수 목록:")
{indent}    keys = [k for k in locals().keys() if not k.startswith('__')]
{indent}    print(keys)
{indent}    print("!"*50 + "\\n")
{indent}    exit()

{indent}print("[Test Mode] Predicting (using X, A, test_indices)...")
{indent}# predict 함수 호출
{indent}preds, _ = model_instance.predict(input_X, input_A, target_idx)

{indent}if target_label is not None:
{indent}    acc = accuracy_score(target_label, preds)
{indent}    print("="*40)
{indent}    print(f"🏆 FINAL TEST ACCURACY: {{acc * 100:.2f}}%")
{indent}    print("="*40)
{indent}else:
{indent}    print("예측 완료 (정답지 없음)")

{indent}exit()
{indent}# --- Test Mode Block End ---
"""
        new_lines.append(code_block)
    else:
        new_lines.append(line)

# 3. 새로운 파일로 저장
with open('gcn_test_only.py', 'w') as f:
    f.writelines(new_lines)

print("✅ 'gcn_test_only.py' 재생성 완료 (변수명 탐색기 탑재)!")