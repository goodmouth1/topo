import sys
import gzip
import pickle
import numpy as np
import scipy.sparse as sp

# ==========================================================
# [긴급 수선] Theano 1.0 vs Lasagne 호환성 문제 해결
try:
    import theano.tensor.signal.pool
    import theano.tensor.signal
    theano.tensor.signal.downsample = theano.tensor.signal.pool
except ImportError:
    pass
# ==========================================================

import theano
import lasagne
from gcnmodel import GraphConv
from gazetteer import Gazetteer
from haversine import haversine

# 재귀 깊이 설정
sys.setrecursionlimit(2000)

class DemoSystem:
    def __init__(self, model_path='./data/model-9475-1.0.pkl', data_path='./data/cmu/dump.pkl'):
        print("\n========================================")
        print("[System] 통합 데모 시스템을 시작합니다.")
        print("========================================")

        self.gaz = Gazetteer()
        
        print(f"\n[System] 사용자 데이터 로딩 및 재조립 중... ({data_path})")
        try:
            with gzip.open(data_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            
            # [수정] 데이터가 흩어져 있어서 하나로 합치는 작업 (Stacking)
            # 순서: Train(5747) -> Val(1825) -> Test(1903)
            self.adj = data[0]         # 전체 친구 관계 (9475명)
            
            # 특징 행렬 합치기 (Train + Val + Test)
            print("   -> 특징 행렬 합치는 중...")
            self.features = sp.vstack((data[1], data[3], data[5]))
            
            self.y_test = data[6]      # 정답 라벨 (Test용)
            self.test_ids = data[9]    # 테스트 유저 ID 목록 (문자열)
            
            # 좌표 변환 지도 로딩 (Item 10=Lat, 11=Lon)
            self.lat_map = data[10]
            self.lon_map = data[11]
            
            # [중요] 테스트 유저가 전체 행렬에서 몇 번째 줄에 있는지 계산
            # (앞에 있는 Train + Val 인원수만큼 건너뛰어야 함)
            self.test_offset = data[1].shape[0] + data[3].shape[0]
            
            print(f"[System] 데이터 로딩 완료! (총 유저: {self.adj.shape[0]}명)")
            
        except Exception as e:
            print(f"[Error] 데이터 로딩 실패: {e}")
            sys.exit(1)

        print(f"[System] AI 모델 조립 및 파라미터 장착 중... ({model_path})")
        try:
            with gzip.open(model_path, 'rb') as f:
                model_params = pickle.load(f, encoding='latin1')
            
            input_size = self.features.shape[1]
            output_size = model_params[-1].shape[0]
            
            print(f"   -> 입력 크기: {input_size}, 출력 클래스: {output_size}")
            
            self.model = GraphConv(
                input_size=input_size,
                output_size=output_size,
                hid_size_list=[300],
                regul_coef=1e-5,
                drop_out=0.5,
                highway=True
            )
            
            print("   -> 모델 구조 컴파일 중... (약 1분 소요)")
            self.model.build_model(self.adj)
            
            lasagne.layers.set_all_param_values(self.model.l_out, model_params)
            print("[System] AI 준비 완료!")
            
        except Exception as e:
            print(f"[Error] 모델 빌드 실패: {e}")
            sys.exit(1)

    def get_coords_from_label(self, label_idx):
        """Class ID(정수)를 위도/경도로 변환"""
        # 딕셔너리 키가 문자열('0')인지 정수(0)인지 확인 후 처리
        key = str(label_idx)
        if key in self.lat_map:
            return (self.lat_map[key], self.lon_map[key])
        return (0.0, 0.0)

    def run_simulation(self, target_word="Springfield", num_samples=3):
        print(f"\n\n>>> 시뮬레이션 시작: 단어 '{target_word}'에 대한 모호성 해소")
        
        # 테스트셋 안에서 랜덤으로 몇 명 뽑기
        random_indices = np.random.choice(len(self.test_ids), num_samples, replace=False)
        
        for i, idx in enumerate(random_indices):
            user_id = self.test_ids[idx]
            
            # [핵심] 전체 행렬에서의 진짜 인덱스 계산
            real_matrix_idx = self.test_offset + idx
            
            print(f"\n---------------- [Test Case {i+1}: {user_id}] ----------------")
            
            try:
                # 1. AI 위치 예측
                # predict 함수는 [인덱스 리스트]를 받음
                preds, _ = self.model.predict(self.features, self.adj, [real_matrix_idx])
                pred_label = preds[0]
                
                # 2. 좌표 변환
                pred_lat, pred_lon = self.get_coords_from_label(pred_label)
                
                # 3. 정답 확인
                true_label = self.y_test[idx]
                true_lat, true_lon = self.get_coords_from_label(true_label)
                
                error_km = haversine((pred_lat, pred_lon), (true_lat, true_lon))
                
                print(f"1. [AI 위치 추적]")
                print(f"   - AI 예측: Class {pred_label} -> ({pred_lat:.4f}, {pred_lon:.4f})")
                print(f"   - 실제 정답: Class {true_label} -> ({true_lat:.4f}, {true_lon:.4f})")
                print(f"   - 오차 거리: 약 {error_km:.2f} km")

                # 4. 지명 모호성 해소
                print(f"2. [지명 모호성 해소: '{target_word}']")
                self.solve_ambiguity(pred_lat, pred_lon, target_word)

            except Exception as e:
                print(f"[Error] 예측 중 오류: {e}")
                import traceback
                traceback.print_exc()

    def solve_ambiguity(self, user_lat, user_lon, mention_text):
        candidates = self.gaz.get_candidates(mention_text, max_results=5)
        if not candidates:
            print("   -> 후보 지명을 찾을 수 없습니다.")
            return

        user_loc = (user_lat, user_lon)
        for c in candidates:
            c['distance_km'] = haversine(user_loc, (c['lat'], c['lon']))

        candidates.sort(key=lambda x: x['distance_km'])
        winner = candidates[0]

        print(f"   -> AI 추천 정답: ★ {winner['name']} ({winner['state']}, {winner['country']}) ★")
        print(f"   -> 근거: 사용자 추정 위치에서 {winner['distance_km']:.2f} km 거리")

if __name__ == "__main__":
    demo = DemoSystem()
    demo.run_simulation(target_word="Springfield", num_samples=3)