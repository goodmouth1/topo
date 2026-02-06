import pickle
import pandas as pd
import numpy as np
import os
import sys
import csv
import json
import re
from haversine import haversine
from openai import OpenAI
from tqdm import tqdm  # 진행률 표시바 (없으면 pip install tqdm)

# ======================================================
# 🔑 [필수] 여기에 OpenAI API 키를 입력하세요!
# ======================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
# ======================================================

# 1. 설정
DATA_DIR = "./data"
DUMP_FILE = os.path.join(DATA_DIR, "dump.pkl")
OUTPUT_FILE = "test_with_ground_truth_llm.pkl"
GEONAMES_URL = "http://download.geonames.org/export/dump/cities1000.zip"

# 모델 설정 (현재 최강 모델)
MODEL_NAME = "gpt-4o"

# 2. GeoNames 로드 (필터링 최소화 - GPT를 믿으니까!)
def load_us_gazetteer():
    print("🌍 [GeoNames] cities1000 로딩 중...")
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
        # 소문자로 변환하여 인덱싱 (GPT가 'paris'라고 줘도 찾을 수 있게)
        name_lower = str(row['asciiname']).lower()
        
        # 너무 짧은 건 여전히 위험하니까 2글자는 제외
        if len(name_lower) < 3: continue 
        
        if name_lower not in gazetteer:
            gazetteer[name_lower] = []
            
        gazetteer[name_lower].append({
            'name': row['asciiname'], # 원래 이름 (대소문자 포함)
            'lat': row['lat'], 
            'lon': row['lon'], 
            'state': row['admin1 code']
        })
        
    print(f"✅ US Gazetteer 로딩 완료! (총 {len(gazetteer)}개 키워드)")
    return gazetteer

# 3. GPT-4o 호출 함수 (핵심!)
client = OpenAI(api_key=OPENAI_API_KEY)

def extract_locations_gpt(text):
    """
    GPT-4o에게 텍스트를 주고 지명 리스트를 JSON으로 받아옴
    """
    system_prompt = """
    You are a precise Named Entity Recognition (NER) system specializing in geolocation.
    Extract ONLY city names or state names from the user's tweet text.
    
    Rules:
    1. Ignore general nouns (e.g., "University", "Park", "Home", "Green").
    2. Ignore person names (e.g., "Jay", "Cleveland Brown").
    3. Ignore fictional places.
    4. Return the result as a JSON object with a key "locations" containing a list of strings.
    5. If no location is found, return {"locations": []}.
    """
    
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            temperature=0, # 창의성 0 (정확하게 뽑기 위해)
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get("locations", [])
    except Exception as e:
        print(f"⚠️ API Error: {e}")
        return []

# 4. 메인 로직
print(f"📦 Loading user data from {DUMP_FILE}...")
try:
    with open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)
except:
    import gzip
    with gzip.open(DUMP_FILE, 'rb') as f:
        data = pickle.load(f)

from data import DataLoader
dl = DataLoader(data_home=DATA_DIR, bucket_size=300, encoding='utf-8')
dl.load_data()
df_test = dl.df_test
us_gazetteer = load_us_gazetteer()

print(f"🤖 GPT-5.2-pro 기반 정답지 생성 시작 (총 {len(df_test)}명 유저)...")
print("💰 주의: API 비용이 발생합니다. (유저당 약 0.1센트 미만)")

ground_truth_data = []

# 테스트를 위해 일단 100명만 돌려볼까요? (전체 다 하려면 [:100] 제거)
# df_target = df_test
df_target = df_test  

for user_id, row in tqdm(df_target.iterrows(), total=len(df_target)):
    text = row['text']
    user_true_lat = row['lat']
    user_true_lon = row['lon']
    
    # 1. GPT가 지명 추출 (소문자/문맥 다 알아서 처리함)
    extracted_names = extract_locations_gpt(text)
    
    if not extracted_names:
        continue
        
    found_toponyms = []
    
    # 2. 추출된 지명을 좌표 후보군(Gazetteer)과 매칭
    for name in extracted_names:
        name_lower = name.lower()
        
        if name_lower in us_gazetteer:
            candidates = us_gazetteer[name_lower]
            
            # 후보가 2개 이상인 경우만 '중의성' 문제로 간주
            if len(candidates) > 1:
                # 거리 계산
                for cand in candidates:
                    cand['dist_from_user'] = haversine((user_true_lat, user_true_lon), (cand['lat'], cand['lon']))
                
                # 거리순 정렬
                candidates.sort(key=lambda x: x['dist_from_user'])
                
                found_toponyms.append({
                    'word': name, # GPT가 뽑아준 원래 철자 (예: Paris)
                    'candidates': candidates,
                    'orig_text': text
                })
    
    if found_toponyms:
        ground_truth_data.append({
            'user_id': user_id,
            'user_true_loc': (user_true_lat, user_true_lon),
            'toponyms': found_toponyms
        })
        
        # 중간 저장 (혹시 에러나면 아까우니까)
        if len(ground_truth_data) % 50 == 0:
             with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(ground_truth_data, f)

# 최종 저장
with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(ground_truth_data, f)

print(f"\n✅ [GPT-5.2-pro] 고품질 정답지 생성 완료! '{OUTPUT_FILE}'")
print(f"📊 유효한 데이터 확보: {len(ground_truth_data)}명")