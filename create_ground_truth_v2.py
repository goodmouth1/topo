import pickle
import pandas as pd
import numpy as np
import os
import sys
import csv
from haversine import haversine
import re

# 1. 설정
DATA_DIR = "./data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
OUTPUT_FILE = "test_with_ground_truth.pkl"
GEONAMES_URL = "http://download.geonames.org/export/dump/cities1000.zip"

# 🚫 노이즈 제거용 블랙리스트 (지명이지만 일반명사로 더 많이 쓰이는 단어들)
BLACKLIST = {
    "University", "College", "School", "North", "South", "East", "West",
    "Green", "White", "Black", "Brown", "New", "Old", "Great", "Little",
    "Grand", "Long", "Short", "High", "Low", "Good", "Bad", "Best",
    "Man", "Woman", "Boy", "Girl", "Job", "Work", "Home", "House",
    "Park", "Lake", "River", "Mountain", "City", "Town", "Village",
    "Spring", "Winter", "Summer", "Fall", "Sun", "Moon", "Star",
    "Reading", "Mobile", "Independence", "Liberty", "Union", "Auburn", "Aurora",
    "Of", "The", "A", "An", "To", "In", "For", "On", "By", "At", "It", "No", "Yes",
    "Love", "Life", "Happy", "One", "Two", "Three"
}

def load_us_gazetteer():
    print("🌍 [GeoNames] cities1000 로딩 및 필터링 중...")
    txt_file = os.path.join(DATA_DIR, "cities1000.txt")
    
    if not os.path.exists(txt_file):
        os.system(f"wget {GEONAMES_URL} -P {DATA_DIR}")
        os.system(f"unzip {DATA_DIR}/cities1000.zip -d {DATA_DIR}")

    cols = ['geonameid', 'name', 'asciiname', 'alternatenames', 'lat', 'lon', 
            'feature class', 'feature code', 'country code', 'cc2', 
            'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code', 
            'population', 'elevation', 'dem', 'timezone', 'modification date']
    
    df = pd.read_csv(txt_file, sep='\t', names=cols, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
    us_df = df[df['country code'] == 'US'].copy()
    
    gazetteer = {}
    for _, row in us_df.iterrows():
        name = str(row['asciiname'])
        
        # 🌟 강력한 필터링 적용
        if len(name) < 3: continue # 2글자 이하 삭제
        if name in BLACKLIST: continue # 금지어 삭제
        
        if name not in gazetteer:
            gazetteer[name] = []
        gazetteer[name].append({
            'lat': row['lat'], 
            'lon': row['lon'], 
            'state': row['admin1 code'],
            'name': name
        })
        
    print(f"✅ 필터링된 US Gazetteer 구축 완료! (총 {len(gazetteer)}개 지명)")
    return gazetteer

print(f"📦 Loading user data from {DUMP_FILE}...")
try:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except:
    import gzip
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

# 원본 텍스트 로딩을 위해 DataLoader 호출
from data import DataLoader
dl = DataLoader(data_home=DATA_DIR, bucket_size=300, encoding='utf-8')
dl.load_data()
df_test = dl.df_test
us_gazetteer = load_us_gazetteer()

print("🎯 정답지 후보 생성 시작...")
ground_truth_data = []

count = 0
for user_id, row in df_test.iterrows():
    text = row['text']
    user_true_lat = row['lat']
    user_true_lon = row['lon']
    
    tokens = text.split()
    found_toponyms = []
    
    for token in tokens:
        # 특수문자 제거 후 대문자 체크를 위해 원본 보존
        clean_token = re.sub(r'[^\w\s]', '', token)
        
        # 🌟 핵심 필터: 첫 글자가 대문자인지 확인 (Proper Noun Check)
        if not clean_token: continue
        if not clean_token[0].isupper(): continue 
        
        if clean_token in us_gazetteer:
            candidates = us_gazetteer[clean_token]
            
            # 후보가 2개 이상이어야 '중의성' 문제가 성립됨
            if len(candidates) > 1:
                # 거리 기반 추천값(참고용) 계산
                for cand in candidates:
                    cand['dist_from_user'] = haversine((user_true_lat, user_true_lon), (cand['lat'], cand['lon']))
                
                candidates.sort(key=lambda x: x['dist_from_user'])

                found_toponyms.append({
                    'word': clean_token,
                    'candidates': candidates,
                    'orig_text': text # 🌟 문맥 확인용 텍스트 저장
                })
    
    if found_toponyms:
        ground_truth_data.append({
            'user_id': user_id,
            'user_true_loc': (user_true_lat, user_true_lon),
            'toponyms': found_toponyms
        })
        count += 1
        if count % 100 == 0:
            print(f"   ... {count}명의 유저 처리 완료")

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(ground_truth_data, f)

print(f"\n✅ 정답지 후보 생성 완료! '{OUTPUT_FILE}'")
print(f"📊 중의성 지명이 발견된 유저 수: {len(ground_truth_data)}명")
