#!/usr/bin/env python
"""
GeoText & Twitter-US bucket_size 튜닝 자동화 스크립트.

GeoText (GPU, 빠름) → Twitter-US (CPU, 느림) 순으로 실행.
결과를 geotextbucket/tuning_results.csv에 저장.

사용법:
  # GeoText만 (빠름, ~30분):
  python -u run_geotext_tuning.py --geotext-only

  # Twitter-US만 (느림, ~수일):
  nohup python -u run_geotext_tuning.py --na-only > na_tuning.log 2>&1 &

  # 전부:
  nohup python -u run_geotext_tuning.py > tuning.log 2>&1 &
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
import argparse
import numpy as np

# 경로 설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MASTER_DIR = os.path.join(BASE_DIR, '..')
GEOTEXT_RAW_DIR = os.path.join(MASTER_DIR, 'data')  # user_info.*.gz가 있는 곳
NA_PKL_GZ = os.path.join(MASTER_DIR, 'data', 'na', 'na.pkl.gz')
GRAVITY_SCRIPT = os.path.join(MASTER_DIR, 'run_gravity_with_gcn_state.py')
GCN_MAIN = os.path.join(MASTER_DIR, 'gcnmain.py')
RESULT_FILE = os.path.join(BASE_DIR, 'tuning_results.csv')
PYTHON = '/home/wang/miniconda3/envs/thesis_gpu/bin/python'

# =============================================
# 실험 설정
# =============================================
GEOTEXT_EXPERIMENTS = [
    # (이름, bucket, hidden, dropout, reg, cel, mindf)
    ('geotext_bucket25',  25,  [300, 300, 300], 0.5, 1e-6, 10, 10),
    ('geotext_bucket50',  50,  [300, 300, 300], 0.5, 1e-6, 10, 10),  # 원논문 설정
    ('geotext_bucket75',  75,  [300, 300, 300], 0.5, 1e-6, 10, 10),
    ('geotext_bucket100', 100, [300, 300, 300], 0.5, 1e-6, 10, 10),
    ('geotext_bucket150', 150, [300, 300, 300], 0.5, 1e-6, 10, 10),
]

NA_EXPERIMENTS = [
    # (이름, bucket, hidden, dropout, reg, cel, mindf)
    ('na_bucket300',  300,  [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket500',  500,  [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket1000', 1000, [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket1500', 1500, [600, 600, 600], 0.5, 1e-6, 10, 10),
    ('na_bucket2400', 2400, [600, 600, 600], 0.5, 1e-6, 10, 10),  # 원논문 설정
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


def evaluate_gravity_geotext(pred_file):
    """
    GeoText용 Gravity Model 평가.
    run_gravity_with_gcn_state.py를 subprocess로 호출하여 결과 파싱.
    """
    result = subprocess.run(
        [PYTHON, GRAVITY_SCRIPT, pred_file],
        capture_output=True, text=True, cwd=MASTER_DIR, timeout=300
    )

    output = result.stdout
    results = {'matched_users': 0}

    # 출력에서 결과 파싱
    for line in output.split('\n'):
        if '매칭된 유저 수' in line:
            try:
                results['matched_users'] = int(line.split(':')[1].strip().replace('명', ''))
            except:
                pass
        if '적중률' in line:
            try:
                # "92.34% (217/235)" 형태 파싱
                pct_str = line.split(':')[1].strip()
                pct = float(pct_str.split('%')[0])
                # 분수 부분
                frac = pct_str.split('(')[1].split(')')[0]
                correct, total = frac.split('/')
                results['gravity_acc'] = pct
                results['gravity_correct'] = int(correct)
                results['gravity_total'] = int(total)
            except:
                pass

    return results


def setup_geotext_data_dir(exp_dir):
    """GeoText raw 파일을 실험 디렉토리에 symlink"""
    os.makedirs(exp_dir, exist_ok=True)

    for split in ['train', 'dev', 'test']:
        src = os.path.join(GEOTEXT_RAW_DIR, f'user_info.{split}.gz')
        dst = os.path.join(exp_dir, f'user_info.{split}.gz')
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)


def run_geotext_experiment(name, bucket, hidden, dropout, reg, cel, mindf):
    """GeoText 실험: GPU + -builddata"""
    print(f"\n{'='*60}")
    print(f"🔬 GeoText 실험: {name}")
    print(f"   bucket={bucket}, hidden={hidden}")
    print(f"{'='*60}")

    exp_dir = os.path.join(BASE_DIR, f'exp_{name}')
    data_dir = os.path.join(exp_dir, 'data')
    setup_geotext_data_dir(data_dir)

    # GCN 훈련 (CPU — cuDNN 미설치로 GPU 불가, GeoText는 소규모라 CPU로 충분)
    hid_args = ' '.join(str(h) for h in hidden)
    cmd = (
        f"THEANO_FLAGS='device=cpu,floatX=float32' "
        f"{PYTHON} -u {GCN_MAIN} "
        f"-d {data_dir} -bucket {bucket} -hid {hid_args} "
        f"-mindf {mindf} -reg {reg} -dropout {dropout} -cel {cel} "
        f"-conv -highway -builddata"
    )
    print(f"   CMD: {cmd}")
    start_time = time.time()

    result = subprocess.run(
        cmd, shell=True, cwd=BASE_DIR,
        capture_output=True, text=True, timeout=3600
    )

    elapsed = time.time() - start_time
    print(f"   훈련 시간: {elapsed/60:.1f}분")

    if result.returncode != 0:
        print(f"   ❌ 훈련 실패!")
        print(f"   stderr (마지막 500자): {result.stderr[-500:]}")
        # stdout에도 에러 정보가 있을 수 있음
        print(f"   stdout (마지막 500자): {result.stdout[-500:]}")
        return None

    # 생성된 예측 파일 찾기 (CWD = BASE_DIR에 저장됨)
    pred_files = sorted(glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')),
                        key=os.path.getmtime)
    if not pred_files:
        print("   ❌ 예측 파일이 생성되지 않음!")
        return None

    pred_file = pred_files[-1]
    pred_name = os.path.basename(pred_file)
    print(f"   예측 파일: {pred_name}")

    # 예측 파일을 exp 디렉토리로 이동
    dest_pred = os.path.join(exp_dir, pred_name)
    shutil.move(pred_file, dest_pred)

    # 다른 남은 gcn 파일도 정리
    for f in glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')):
        shutil.move(f, os.path.join(exp_dir, os.path.basename(f)))

    # GCN 위치 예측 평가
    print("   📊 GCN 위치 예측 평가...")
    gcn_metrics = evaluate_gcn_location(dest_pred)
    print(f"      Mean: {gcn_metrics['mean_km']}km, Median: {gcn_metrics['median_km']}km, "
          f"Acc@161: {gcn_metrics['acc_161']}% ({gcn_metrics['n_users']}명)")

    # Gravity 평가 (예측 파일을 master_thesis/ 디렉토리에 임시 복사)
    print("   🌍 Gravity Model 평가...")
    temp_pred = os.path.join(MASTER_DIR, f'_temp_pred_{name}.pkl')
    shutil.copy2(dest_pred, temp_pred)
    try:
        gravity_metrics = evaluate_gravity_geotext(temp_pred)
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
        'dataset': 'GeoText',
        'name': name,
        'bucket': bucket,
        'hidden': str(hidden),
        'train_time_min': round(elapsed / 60, 1),
        **gcn_metrics,
        **gravity_metrics,
    }


def run_na_experiment(name, bucket, hidden, dropout, reg, cel, mindf):
    """Twitter-US 실험: CPU + 클래스 재할당"""
    print(f"\n{'='*60}")
    print(f"🔬 Twitter-US 실험: {name}")
    print(f"   bucket={bucket}, hidden={hidden}")
    print(f"{'='*60}")

    exp_dir = os.path.join(BASE_DIR, f'exp_{name}')
    data_dir = os.path.join(exp_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)

    # 1단계: 클래스 재할당 (na.pkl.gz → 새 dump.pkl)
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

    # 2단계: GCN 훈련 (CPU, dump.pkl에서 로드 — -builddata 없이)
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
        timeout=86400  # 24시간 타임아웃
    )

    elapsed = time.time() - start_time
    print(f"   훈련 시간: {elapsed/60:.1f}분 ({elapsed/3600:.1f}시간)")

    if result.returncode != 0:
        print(f"   ❌ 훈련 실패!")
        print(f"   stderr (마지막 500자): {result.stderr[-500:]}")
        print(f"   stdout (마지막 500자): {result.stdout[-500:]}")
        return None

    # 생성된 예측 파일 찾기
    pred_files = sorted(glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')),
                        key=os.path.getmtime)
    if not pred_files:
        print("   ❌ 예측 파일이 생성되지 않음!")
        return None

    pred_file = pred_files[-1]
    pred_name = os.path.basename(pred_file)
    print(f"   예측 파일: {pred_name}")

    # 예측 파일을 exp 디렉토리로 이동
    dest_pred = os.path.join(exp_dir, pred_name)
    shutil.move(pred_file, dest_pred)

    for f in glob.glob(os.path.join(BASE_DIR, 'gcn_*.pkl')):
        shutil.move(f, os.path.join(exp_dir, os.path.basename(f)))

    # GCN 위치 예측 평가
    print("   📊 GCN 위치 예측 평가...")
    gcn_metrics = evaluate_gcn_location(dest_pred)
    print(f"      Mean: {gcn_metrics['mean_km']}km, Median: {gcn_metrics['median_km']}km, "
          f"Acc@161: {gcn_metrics['acc_161']}% ({gcn_metrics['n_users']}명)")

    # Gravity 평가
    print("   🌍 Gravity Model 평가...")
    temp_pred = os.path.join(MASTER_DIR, f'_temp_pred_{name}.pkl')
    shutil.copy2(dest_pred, temp_pred)
    try:
        gravity_metrics = evaluate_gravity_geotext(temp_pred)
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


def save_results(all_results):
    """결과를 CSV로 저장"""
    if not all_results:
        print("결과 없음!")
        return

    fieldnames = list(all_results[0].keys())
    with open(RESULT_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"\n💾 결과 저장: {RESULT_FILE}")


def print_summary(all_results):
    """결과 요약 출력"""
    if not all_results:
        return

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GeoText & Twitter-US 튜닝')
    parser.add_argument('--geotext-only', action='store_true', help='GeoText만 실행')
    parser.add_argument('--na-only', action='store_true', help='Twitter-US만 실행')
    args = parser.parse_args()

    run_geotext = not args.na_only
    run_na = not args.geotext_only

    print("=" * 60)
    print("GeoText & Twitter-US Bucket Size 튜닝")
    if run_geotext:
        print(f"  GeoText 실험: {len(GEOTEXT_EXPERIMENTS)}개 (GPU)")
    if run_na:
        print(f"  Twitter-US 실험: {len(NA_EXPERIMENTS)}개 (CPU)")
    print(f"  결과 저장: {RESULT_FILE}")
    print("=" * 60)

    all_results = []

    # GeoText 실험 (GPU, 빠름)
    if run_geotext:
        print("\n" + "=" * 60)
        print("📌 Phase 1: GeoText 실험 (GPU)")
        print("=" * 60)

        for exp in GEOTEXT_EXPERIMENTS:
            try:
                row = run_geotext_experiment(*exp)
                if row:
                    all_results.append(row)
                    save_results(all_results)  # 매 실험 후 저장 (중간 중단 대비)
            except Exception as e:
                print(f"   ❌ 에러: {e}")
                import traceback
                traceback.print_exc()

    # Twitter-US 실험 (CPU, 느림)
    if run_na:
        print("\n" + "=" * 60)
        print("📌 Phase 2: Twitter-US 실험 (CPU)")
        print("=" * 60)

        # 기존 GeoText 결과 로드 (있으면)
        if os.path.exists(RESULT_FILE) and not run_geotext:
            import csv as csv_mod
            with open(RESULT_FILE, 'r') as f:
                reader = csv_mod.DictReader(f)
                for row in reader:
                    # 숫자 필드 변환
                    for key in ['bucket', 'mean_km', 'median_km', 'acc_161', 'n_users',
                                'matched_users', 'gravity_acc', 'gravity_correct',
                                'gravity_total', 'train_time_min']:
                        if key in row and row[key]:
                            try:
                                row[key] = float(row[key])
                            except:
                                pass
                    all_results.append(row)

        for exp in NA_EXPERIMENTS:
            try:
                row = run_na_experiment(*exp)
                if row:
                    all_results.append(row)
                    save_results(all_results)  # 매 실험 후 저장
            except Exception as e:
                print(f"   ❌ 에러: {e}")
                import traceback
                traceback.print_exc()

    # 최종 결과
    save_results(all_results)
    print_summary(all_results)

    print(f"\n✅ 튜닝 완료! 결과: {RESULT_FILE}")
