#!/usr/bin/env python
"""
Twitter-US (na.pkl.gz)의 클래스를 새로운 bucket_size로 재할당하는 유틸리티.

na.pkl.gz에는 raw user_info 파일이 없으므로,
기존 TF-IDF와 인접행렬은 유지하고 KD-Tree 클래스만 재할당한다.

사용법:
  python reassign_classes.py --bucket 500 --output ./na_bucket500/
"""

import os
import sys
import pickle
import gzip
import argparse
import numpy as np
from collections import OrderedDict

# kdtree는 상위 디렉토리에 있음
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import kdtree
from sklearn.neighbors import NearestNeighbors


def haversine_np(lat1, lon1, lat2, lon2):
    """numpy용 haversine 거리 (km)"""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def haversine_sklearn(u, v):
    """sklearn NearestNeighbors 메트릭용"""
    lat1, lon1 = u
    lat2, lon2 = v
    return haversine_np(lat1, lon1, lat2, lon2)


def reassign(input_pkl_gz, bucket_size, output_dir):
    """
    pkl.gz 파일을 로드하고 새로운 bucket_size로 클래스를 재할당한 후 dump.pkl로 저장

    pkl.gz 구조 (13개 요소 tuple):
      (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test,
       U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"📂 로딩: {input_pkl_gz}")
    with gzip.open(input_pkl_gz, 'rb') as f:
        data = pickle.load(f, encoding='latin1')

    A, X_train, Y_train_old, X_dev, Y_dev_old, X_test, Y_test_old, \
        U_train, U_dev, U_test, classLatMedian_old, classLonMedian_old, userLocation = data

    print(f"  유저: train={len(U_train)}, dev={len(U_dev)}, test={len(U_test)}")
    print(f"  기존 클래스 수: {len(classLatMedian_old)}")

    # userLocation에서 train/dev/test 좌표 추출
    # userLocation의 값은 "lat,lon" 문자열 또는 (lat, lon) 튜플
    def parse_loc(loc):
        if isinstance(loc, str):
            parts = loc.split(',')
            return float(parts[0]), float(parts[1])
        elif isinstance(loc, (tuple, list)):
            return float(loc[0]), float(loc[1])
        else:
            return float(loc[0]), float(loc[1])

    train_locs = np.array([parse_loc(userLocation[u]) for u in U_train])
    dev_locs = np.array([parse_loc(userLocation[u]) for u in U_dev])
    test_locs = np.array([parse_loc(userLocation[u]) for u in U_test])

    print(f"  좌표 추출 완료: train={train_locs.shape}, dev={dev_locs.shape}, test={test_locs.shape}")

    # KD-Tree 클러스터링 (새 bucket_size)
    print(f"  KD-Tree 클러스터링 (bucket_size={bucket_size})...")
    clusterer = kdtree.KDTreeClustering(bucket_size=bucket_size)
    clusterer.fit(train_locs)
    train_clusters = clusterer.get_clusters()

    # 클래스별 중앙값 계산
    from collections import defaultdict
    cluster_points = defaultdict(list)
    for i, cluster in enumerate(train_clusters):
        cluster_points[cluster].append(train_locs[i])

    cluster_median = OrderedDict()
    for cluster in sorted(cluster_points):
        points = cluster_points[cluster]
        median_lat = np.median([p[0] for p in points])
        median_lon = np.median([p[1] for p in points])
        cluster_median[cluster] = (median_lat, median_lon)

    num_classes = len(cluster_median)
    print(f"  새 클래스 수: {num_classes}")

    # Dev/Test 유저를 가장 가까운 클래스에 할당
    print(f"  Dev/Test 유저 클래스 할당...")
    median_coords = np.array([v for v in cluster_median.values()])
    nnbr = NearestNeighbors(n_neighbors=1, algorithm='brute', leaf_size=1,
                            metric=haversine_sklearn, n_jobs=4)
    nnbr.fit(median_coords)

    dev_classes = nnbr.kneighbors(dev_locs, n_neighbors=1, return_distance=False)[:, 0]
    test_classes = nnbr.kneighbors(test_locs, n_neighbors=1, return_distance=False)[:, 0]

    # 새 classLatMedian, classLonMedian
    new_classLatMedian = {str(c): cluster_median[c][0] for c in cluster_median}
    new_classLonMedian = {str(c): cluster_median[c][1] for c in cluster_median}

    # 새 데이터 tuple 구성 (A, X는 그대로 유지)
    new_data = (A, X_train, train_clusters, X_dev, dev_classes, X_test, test_classes,
                U_train, U_dev, U_test, new_classLatMedian, new_classLonMedian, userLocation)

    # dump.pkl로 저장 (gzip 압축 — gcnmain.py의 load_obj가 gzip.open을 사용)
    output_file = os.path.join(output_dir, 'dump.pkl')
    print(f"  저장: {output_file}")
    with gzip.open(output_file, 'wb') as f:
        pickle.dump(new_data, f)

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  ✅ 완료! ({size_mb:.1f} MB, {num_classes} 클래스)")

    return num_classes


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Twitter-US 클래스 재할당')
    parser.add_argument('--input', default='../data/na/na.pkl.gz', help='입력 pkl.gz 파일')
    parser.add_argument('--bucket', type=int, required=True, help='새 bucket_size')
    parser.add_argument('--output', required=True, help='출력 디렉토리')

    args = parser.parse_args()
    reassign(args.input, args.bucket, args.output)
