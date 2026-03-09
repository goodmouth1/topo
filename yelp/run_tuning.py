#!/usr/bin/env python
"""
하이퍼파라미터 튜닝 자동화 스크립트.
여러 설정으로 GCN 훈련 → Gravity Model 평가를 순차 실행하고
결과를 tuning_results.csv에 모아준다.

사용법:
  nohup python -u run_tuning.py > tuning.log 2>&1 &
  tail -f tuning.log
"""

import subprocess
import sys
import os
import json
import csv
import time
import glob
import pickle
import gzip
import numpy as np

YELP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(YELP_DIR, 'yelp_geotext_format')
RESULT_FILE = os.path.join(YELP_DIR, 'tuning_results.csv')

# =============================================
# 실험 설정 (bucket_size=300은 이미 완료)
# =============================================
EXPERIMENTS = [
    # (이름, bucket, hidden, mindf, dropout, reg, cel)
    ('bucket100_h300x3', 100, [300, 300, 300], 10, 0.5, 1e-6, 10),
    ('bucket200_h300x3', 200, [300, 300, 300], 10, 0.5, 1e-6, 10),
    # bucket300은 이미 완료 → 결과만 수집
    ('bucket500_h300x3', 500, [300, 300, 300], 10, 0.5, 1e-6, 10),
    ('bucket200_h500x3', 200, [500, 500, 500], 10, 0.5, 1e-6, 10),
    ('bucket200_h300x2', 200, [300, 300],      10, 0.5, 1e-6, 10),
]


def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def evaluate_gcn_location(pred_file):
    """GCN 위치 예측 정확도 (Mean, Median, Acc@161km)"""
    with open(pred_file, 'rb') as f:
        distances, latlon_true, latlon_pred = pickle.load(f)

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
    }


def evaluate_gravity(pred_file, eval_data_file):
    """Gravity Model 평가 (다양한 beta)"""
    with open(pred_file, 'rb') as f:
        distances, latlon_true, latlon_pred = pickle.load(f)

    # test 유저 순서 매칭
    test_users = []
    test_file = os.path.join(DATA_DIR, 'user_info.test.gz')
    with gzip.open(test_file, 'rt') as f:
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                test_users.append(parts[0])

    user_gcn_pred = {}
    for i, uid in enumerate(test_users):
        if i < len(latlon_pred):
            user_gcn_pred[uid] = latlon_pred[i]

    with open(eval_data_file, 'rb') as f:
        eval_data = pickle.load(f)

    results = {}

    # Baseline: Population Only
    correct_pop = 0
    total_pop = 0
    for item in eval_data:
        candidates = item['candidates']
        true_idx = item['true_candidate_idx']
        best_idx = max(range(len(candidates)), key=lambda i: candidates[i]['population'])
        if best_idx == true_idx:
            correct_pop += 1
        total_pop += 1
    results['baseline_pop'] = round(correct_pop / total_pop * 100, 2) if total_pop > 0 else 0

    # Gravity Model (beta별)
    for beta in [0.5, 1.0, 1.5, 2.0]:
        correct = 0
        total = 0
        for item in eval_data:
            uid = item['user_id']
            if uid not in user_gcn_pred:
                continue
            pred_lat, pred_lon = user_gcn_pred[uid]
            candidates = item['candidates']
            true_idx = item['true_candidate_idx']

            best_score = -1
            best_idx = -1
            for idx, cand in enumerate(candidates):
                dist = haversine(pred_lat, pred_lon, cand['lat'], cand['lon'])
                if dist < 1:
                    dist = 1
                pop = max(cand['population'], 1000)
                score = pop / (dist ** beta)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            if best_idx == true_idx:
                correct += 1
            total += 1

        results[f'gravity_b{beta}'] = round(correct / total * 100, 2) if total > 0 else 0

    return results


def run_experiment(name, bucket, hidden, mindf, dropout, reg, cel):
    """GCN 훈련 → 평가 한 세트 실행"""
    print(f"\n{'='*60}")
    print(f"실험: {name}")
    print(f"  bucket={bucket}, hidden={hidden}, mindf={mindf}, dropout={dropout}, reg={reg}, cel={cel}")
    print(f"{'='*60}")

    # 출력 디렉토리
    exp_dir = os.path.join(YELP_DIR, f'exp_{name}')
    os.makedirs(exp_dir, exist_ok=True)

    # 기존 dump 캐시 삭제 (bucket 변경 시 필요)
    for cache in glob.glob(os.path.join(DATA_DIR, 'dump*.pkl*')):
        os.remove(cache)
        print(f"  캐시 삭제: {cache}")

    # GCN 훈련
    hid_args = ' '.join(str(h) for h in hidden)
    cmd = (
        f"python -u {os.path.join(YELP_DIR, 'yelp_gcnmain.py')} "
        f"-d {DATA_DIR} -bucket {bucket} -hid {hid_args} "
        f"-mindf {mindf} -reg {reg} -dropout {dropout} -cel {cel} "
        f"-conv -highway -builddata"
    )
    print(f"  명령어: {cmd}")
    start_time = time.time()

    result = subprocess.run(cmd, shell=True, cwd=YELP_DIR,
                           capture_output=True, text=True, timeout=14400)  # 4시간 타임아웃

    elapsed = time.time() - start_time
    print(f"  훈련 시간: {elapsed/60:.1f}분")

    if result.returncode != 0:
        print(f"  ❌ 훈련 실패!")
        print(f"  stderr: {result.stderr[-500:]}")
        return None

    # 생성된 pkl 파일 찾기
    pred_files = sorted(glob.glob(os.path.join(YELP_DIR, 'gcn_*_test.pkl')), key=os.path.getmtime)
    if not pred_files:
        print("  ❌ 예측 파일이 생성되지 않음!")
        return None

    test_pred = pred_files[-1]  # 가장 최근 파일
    print(f"  예측 파일: {test_pred}")

    # 예측 파일을 exp 디렉토리로 복사
    import shutil
    dest = os.path.join(exp_dir, os.path.basename(test_pred))
    shutil.copy2(test_pred, dest)

    # dev 예측도 복사
    dev_preds = sorted(glob.glob(os.path.join(YELP_DIR, 'gcn_*.pkl')), key=os.path.getmtime)
    for dp in dev_preds:
        if '_test' not in dp:
            shutil.copy2(dp, os.path.join(exp_dir, os.path.basename(dp)))

    # 평가
    eval_data_file = os.path.join(YELP_DIR, 'yelp_eval_data.pkl')

    print("  GCN 위치 예측 평가...")
    gcn_metrics = evaluate_gcn_location(test_pred)
    print(f"    Mean: {gcn_metrics['mean_km']}km, Median: {gcn_metrics['median_km']}km, Acc@161: {gcn_metrics['acc_161']}%")

    print("  Gravity Model 평가...")
    gravity_metrics = evaluate_gravity(test_pred, eval_data_file)
    print(f"    Baseline(Pop): {gravity_metrics['baseline_pop']}%")
    for k, v in gravity_metrics.items():
        if k.startswith('gravity'):
            print(f"    {k}: {v}%")

    # 결과 종합
    row = {
        'name': name,
        'bucket': bucket,
        'hidden': str(hidden),
        'mindf': mindf,
        'dropout': dropout,
        'reg': reg,
        'cel': cel,
        'train_time_min': round(elapsed / 60, 1),
        **gcn_metrics,
        **gravity_metrics,
        'best_gravity': max(v for k, v in gravity_metrics.items() if k.startswith('gravity')),
        'improvement': round(max(v for k, v in gravity_metrics.items() if k.startswith('gravity')) - gravity_metrics['baseline_pop'], 2),
    }

    return row


def collect_existing_result():
    """이미 완료된 bucket=300 결과 수집"""
    test_pred = os.path.join(YELP_DIR, 'gcn_1.0_percent_pred_1030_test.pkl')
    eval_data_file = os.path.join(YELP_DIR, 'yelp_eval_data.pkl')

    if not os.path.exists(test_pred) or not os.path.exists(eval_data_file):
        return None

    print("\n기존 bucket=300 결과 수집...")
    gcn_metrics = evaluate_gcn_location(test_pred)
    gravity_metrics = evaluate_gravity(test_pred, eval_data_file)

    return {
        'name': 'bucket300_h300x3 (기존)',
        'bucket': 300,
        'hidden': '[300, 300, 300]',
        'mindf': 10,
        'dropout': 0.5,
        'reg': 1e-6,
        'cel': 10,
        'train_time_min': 'N/A',
        **gcn_metrics,
        **gravity_metrics,
        'best_gravity': max(v for k, v in gravity_metrics.items() if k.startswith('gravity')),
        'improvement': round(max(v for k, v in gravity_metrics.items() if k.startswith('gravity')) - gravity_metrics['baseline_pop'], 2),
    }


if __name__ == '__main__':
    print("=" * 60)
    print("Yelp GCN + Gravity Model 하이퍼파라미터 튜닝")
    print(f"실험 수: {len(EXPERIMENTS)} + 기존 1개")
    print("=" * 60)

    all_results = []

    # 기존 결과 수집
    existing = collect_existing_result()
    if existing:
        all_results.append(existing)
        print(f"  bucket=300 기존 결과: best_gravity={existing['best_gravity']}%, improvement={existing['improvement']}%p")

    # 새 실험 실행
    for exp in EXPERIMENTS:
        try:
            row = run_experiment(*exp)
            if row:
                all_results.append(row)
        except Exception as e:
            print(f"  ❌ 실험 {exp[0]} 에러: {e}")

    # 결과 CSV 저장
    if all_results:
        fieldnames = all_results[0].keys()
        with open(RESULT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\n{'='*60}")
        print(f"전체 결과 저장: {RESULT_FILE}")
        print(f"{'='*60}")

        # 결과 요약 출력
        print(f"\n{'이름':<30} {'bucket':>6} {'hidden':<16} {'Median':>8} {'Acc@161':>8} {'Best Grav':>10} {'개선폭':>8}")
        print("-" * 90)
        for r in sorted(all_results, key=lambda x: -x.get('best_gravity', 0)):
            print(f"{r['name']:<30} {r['bucket']:>6} {r['hidden']:<16} {r['median_km']:>7}km {r['acc_161']:>7}% {r['best_gravity']:>9}% {r['improvement']:>7}%p")
    else:
        print("결과 없음!")
