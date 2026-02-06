import pandas as pd
import os

# 데이터 파일 경로 (다운로드 받은 파일 이름과 같아야 함)
DATA_PATH = '/Users/coys/Library/CloudStorage/GoogleDrive-wangfubu@gmail.com/내 드라이브/realu/master_thesis/data/allCountries.txt'

class Gazetteer:
    def __init__(self, path=DATA_PATH):
        print(f"[Gazetteer] Loading GeoNames data from {path}...")
        print("[Gazetteer] This requires loading 1.5GB into memory. Please wait...")
        
        # GeoNames 컬럼 정의
        columns = [
            'geonameid', 'name', 'asciiname', 'alternatenames', 
            'lat', 'lon', 'feature_class', 'feature_code', 
            'country_code', 'cc2', 'admin1_code', 'admin2_code', 
            'admin3_code', 'admin4_code', 'population', 
            'elevation', 'dem', 'timezone', 'modification_date'
        ]
        
        try:
            # 1. 덩어리(Chunk) 단위로 읽어서 필요한 것만 골라냄
            # feature_class 'P'는 사람이 사는 곳(City, Village)을 의미
            chunks = []
            chunk_size = 500000 # 50만 줄씩 읽기
            
            # 필요한 컬럼만 지정해서 메모리 절약
            use_cols = ['name', 'asciiname', 'lat', 'lon', 'feature_class', 'country_code', 'admin1_code', 'population']
            
            reader = pd.read_csv(
                path, sep='\t', header=None, names=columns, usecols=use_cols,
                dtype={'admin1_code': str, 'population': float},
                encoding='utf-8', quoting=3, chunksize=chunk_size, on_bad_lines='skip'
            )

            for i, chunk in enumerate(reader):
                # 사람이 사는 곳(P)만 필터링
                mask = chunk['feature_class'] == 'P'
                chunks.append(chunk[mask])
                
                # 진행 상황 표시 (선택사항)
                if i % 10 == 0:
                    print(f"  - Processing chunk {i}...")

            # 하나로 합치기
            self.df = pd.concat(chunks, ignore_index=True)
            
            # 검색 속도를 위해 소문자 컬럼 미리 생성
            self.df['name_lower'] = self.df['name'].str.lower()
            self.df['ascii_lower'] = self.df['asciiname'].str.lower()
            
            # 인구수 많은 순서로 정렬 (유명한 도시가 먼저 나오게)
            self.df.sort_values(by='population', ascending=False, inplace=True)
            
            print(f"[Gazetteer] Loaded {len(self.df)} populated places successfully!")
            
        except FileNotFoundError:
            print(f"[Error] {path} 파일을 찾을 수 없습니다.")
            print("data 폴더에 allCountries.txt 파일을 넣어주세요.")
            self.df = None

    def get_candidates(self, toponym, max_results=5):
        """
        지명(toponym)을 입력하면 후보 좌표 리스트를 반환
        """
        if self.df is None:
            return []
            
        search = toponym.lower()
        
        # 이름 일치 검색 (원본 이름 or 영어 이름)
        mask = (self.df['name_lower'] == search) | (self.df['ascii_lower'] == search)
        candidates = self.df[mask].head(max_results) # 상위 N개만 추출
        
        results = []
        for _, row in candidates.iterrows():
            results.append({
                "name": row['name'],
                "country": row['country_code'],
                "state": row['admin1_code'],
                "lat": row['lat'],
                "lon": row['lon'],
                "pop": row['population']
            })
            
        return results

# --- 테스트 코드 ---
if __name__ == "__main__":
    # 데이터 로딩 테스트
    gaz = Gazetteer()
    
    # 검색 테스트
    target = "Springfield"
    print(f"\nSearching for '{target}'...")
    candidates = gaz.get_candidates(target, max_results=10)
    
    for i, c in enumerate(candidates):
        print(f"{i+1}. {c['name']} ({c['state']}, {c['country']}) - Pop: {c['pop']} -> ({c['lat']}, {c['lon']})")