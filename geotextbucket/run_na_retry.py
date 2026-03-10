#!/usr/bin/env python
"""
타임아웃으로 실패한 Twitter-US 실험 3개만 재실행.
(na_bucket300, na_bucket1000, na_bucket1500)

사용법:
  nohup /home/wang/miniconda3/envs/thesis_gpu/bin/python -u run_na_retry.py > na_retry.log 2>&1 &
  tail -f na_retry.log
"""

import subprocess
import sys
import os
import csv
import time
import glob
import pickle
import gzip
import shutil
import numpy as np

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(BASE_DIR, '..')
NA_PKL_GZ = os.path.join(MASTER_DIR, 'data', 'na', 'na.pkl.gz')
GRAVITY_SCRIPT = os.path.join(MASTER_DIR, 'run_gravity_with_gcn_state.py')
GCN_MAIN = os.path.join(MASTER_DIR, 'gcnmain.py')
RESULT_FILE = os.path.join(BASE_DIR, 'tuning_results.csv')
PYTHON = '/home/wang/miniconda3/envs/thesis_gpu/bin/python'

# 실패한 3개만
RETRY_EXPERIMENTS = [
    ('na_bucket300',  300,  [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket1000', 1000, [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket1500', 1500, [600, 600, 600], 0.5, 1e-6, 10, 10),
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def evaluate_gcn_location(pred_file):
    with open(pred_file, 'rb') as f:
        distances, latlon_true, latlon_pred = pickle.load(f, encoding='latin1')
    true_arr = np.array(latlon_true)
    pred_arr = np.array(latlon_pred)
    dists = []
    for i in range(len(true_arr)):
        d = haversine(true_arr[i][0], true_arr[i][1], pred_arr[i][0], pred_arr[i][1])
        dists.append(d)
    dists = np.array(dists)
    return {
        'mean_km': round(np.mean(dists), 1),
        'median_km': round(np.median(dists), 1),
        'acc_161': round(100 * np.sum(dists < 161) / len(dists), 2),
        'n_users': len(dists),
    }


def evaluate_gravity(pred_file):
    result = subprocess.run(
        [PYTHON, GRAVITY_SCRIPT, pred_file],
        capture_output=True, text=True, cwd=MASTER_DIR, timeout=300
    )
    output = result.stdout
    results = {'matched_users': 0}
    for line in output.split('\n'):
        if '매칭된 유저 수' in line:
            try:
                results['matched_users'] = int(line.split(':')[1].strip().replace('명', ''))
            except:
                pass
        if '적중률' in line:
            try:
                pct_str = line.split(':')[1].strip()
                pct = float(pct_str.split('%')[0])
                frac = pct_str.split('(')[1].split(')')[0]
                correct, total = frac.split('/')
                results['gravity_acc'] = pct
                results['gravity_correct'] = int(correct)
                results['gravity_total'] = int(total)
            except:
                pass
    return results


def run_experiment(name, bucket, hidden, dropout, reg, cel, mindf):
    print(f"\n{'='*60}")
    print(f"🔬 Twitter-US 실험: {name}")
    print(f"   bucket={bucket}, hidden={hidden}")
    print(f"{'='*60}")

    exp_dir = os.path.join(BASE_DIR, f'exp_{name}')
    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1단계: 클래스 재할당 (타임아웃 3600초 = 1시간)
    print("   📦 클래스 재할당 중...")
    reassign_cmd = (
        f"{PYTHON} -u {os.path.join(BASE_DIR, 'reassign_classes.py')} "
        f"--input {NA_PKL_GZ} --bucket {bucket} --output {data_dir}"
    )
    result = subprocess.run(reassign_cmd, shell=True, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"   ❌ 클래스 재할당 실패!")
        print(f"   stderr: {result.stderr[-500:]}")
        return None
    print(result.stdout)

    # 2단계: GCN 훈련
    hid_args = ' '.join(str(h) for h in hidden)
    cmd = (
        f"THEANO_FLAGS='device=cpu,floatX=float32' "
        f"{PYTHON} -u {GCN_MAIN} "
        f"-d {data_dir} -hid {hid_args} "
        f"-mindf {mindf} -reg {reg} -dropout {dropout} -cel {cel} "
        f"-conv -highway"
    )
    print(f"   CMD: {cmd}")
    start_time = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=BASE_DIR,
        capture_output=True, text=True,
        timeout=86400
    )

    elapsed = time.time() - start_time
    print(f"   훈련 시간: {elapsed/60:.1f}분 ({elapsed/3600:.1f}시간)")

    if result.returncode != 0:
        print(f"   ❌ 훈련 실패!")
        print(f"   stderr (마지막 500자): {result.stderr[-500:]}")
        print(f"   stdout (마지막 500자): {result.stdout[-500:]}")
        return None

    # 예측 파일 찾기
    pred_files = sorted(glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')),
                        key=os.path.getmtime)
    if not pred_files:
        print("   ❌ 예측 파일이 생성되지 않음!")
        return None

    pred_file = pred_files[-1]
    pred_name = os.path.basename(pred_file)
    print(f"   예측 파일: {pred_name}")

    dest_pred = os.path.join(exp_dir, pred_name)
    shutil.move(pred_file, dest_pred)
    for f in glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')):
        shutil.move(f, os.path.join(exp_dir, os.path.basename(f)))

    # 평가
    print("   📊 GCN 위치 예측 평가...")
    gcn_metrics = evaluate_gcn_location(dest_pred)
    print(f"      Mean: {gcn_metrics['mean_km']}km, Median: {gcn_metrics['median_km']}km, "
          f"Acc@161: {gcn_metrics['acc_161']}% ({gcn_metrics['n_users']}명)")

    print("   🌍 Gravity Model 평가...")
    temp_pred = os.path.join(MASTER_DIR, f'_temp_pred_{name}.pkl')
    shutil.copy2(dest_pred, temp_pred)
    try:
        gravity_metrics = evaluate_gravity(temp_pred)
        if 'gravity_acc' in gravity_metrics:
            print(f"      매칭: {gravity_metrics['matched_users']}명, "
                  f"정확도: {gravity_metrics['gravity_acc']}% "
                  f"({gravity_metrics['gravity_correct']}/{gravity_metrics['gravity_total']})")
        else:
            print(f"      ⚠️ Gravity 결과 파싱 실패")
            gravity_metrics = {'gravity_acc': 0, 'gravity_correct': 0, 'gravity_total': 0, 'matched_users': 0}
    finally:
        if os.path.exists(temp_pred):
            os.remove(temp_pred)

    return {
        'dataset': 'Twitter-US',
        'name': name,
        'bucket': bucket,
        'hidden': str(hidden),
        'train_time_min': round(elapsed / 60, 1),
        **gcn_metrics,
        **gravity_metrics,
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Twitter-US 재실행 (실패한 3개: bucket300, 1000, 1500)")
    print(f"  타임아웃: 3600초 (1시간)")
    print("=" * 60)

    # 기존 CSV 로드
    existing_results = []
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key in ['bucket', 'mean_km', 'median_km', 'acc_161', 'n_users',
                            'matched_users', 'gravity_acc', 'gravity_correct',
                            'gravity_total', 'train_time_min']:
                    if key in row and row[key]:
                        try:
                            row[key] = float(row[key])
                        except:
                            pass
                existing_results.append(row)
        print(f"  기존 결과 {len(existing_results)}개 로드")

    new_results = []
    for exp in RETRY_EXPERIMENTS:
        try:
            row = run_experiment(*exp)
            if row:
                new_results.append(row)
                # 매 실험 후 즉시 CSV 저장 (기존 + 새 결과)
                all_results = existing_results + new_results
                fieldnames = list(all_results[0].keys())
                with open(RESULT_FILE, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(all_results)
                print(f"   💾 결과 저장 완료 (총 {len(all_results)}개)")
        except Exception as e:
            print(f"   ❌ 에러: {e}")
            import traceback
            traceback.print_exc()

    # 최종 요약
    all_results = existing_results + new_results
    print(f"\n{'='*90}")
    print(f"📊 전체 결과 요약")
    print(f"{'='*90}")
    print(f"{'데이터셋':<12} {'이름':<22} {'bucket':>6} {'Median':>8} {'Acc@161':>8} "
          f"{'매칭':>5} {'Gravity':>8} {'(correct/total)':>16}")
    print("-" * 90)
    for r in all_results:
        grav = f"{r.get('gravity_acc', 'N/A')}%"
        frac = f"({r.get('gravity_correct', '?')}/{r.get('gravity_total', '?')})"
        print(f"{r['dataset']:<12} {r['name']:<22} {r['bucket']:>6} "
              f"{r['median_km']:>7}km {r['acc_161']:>7}% "
              f"{r.get('matched_users', 0):>5} {grav:>8} {frac:>16}")

    print(f"\n✅ 재실행 완료! 결과: {RESULT_FILE}")
