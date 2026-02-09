import pickle
import os
import sys
import pandas as pd
import csv
import re
from haversine import haversine
from openai import OpenAI

# ======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    try:
        import getpass
        OPENAI_API_KEY = getpass.getpass("🔑 OpenAI API 키를 입력하세요 (Hidden): ").strip()
    except:
        OPENAI_API_KEY = input("🔑 OpenAI API 키를 입력하세요: ").strip()

if not OPENAI_API_KEY:
    print("❌ API Key가 입력되지 않았습니다. 프로그램을 종료합니다.")
    sys.exit() 
# ======================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# 설정 및 색상
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_FILES = [
    os.path.join(BASE_DIR, "test_with_ground_truth_llm.pkl"),
    os.path.join(BASE_DIR, "test_with_ground_truth.pkl")
]
OUTPUT_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "manual_labeled_data.pkl")
DATA_DIR = os.path.join(BASE_DIR, "data")
GEONAMES_FILE = os.path.join(DATA_DIR, "cities1000.txt")

YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'

# 1. 번역 함수 (GPT-4o-mini)
def translate_to_korean(text):
    """트윗을 한국어로 번역 (슬랭/문맥 고려)"""
    try:
        # 너무 짧으면 번역 스킵
        if len(text) < 5: return text
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a translator who translates informal English tweets into natural Korean. Maintain the slang/tone. Keep location names in English or pronounce them naturally."},
                {"role": "user", "content": f"Translate this tweet to Korean:\n{text}"}
            ],
            temperature=0.3,
            max_tokens=10000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"(번역 실패: {e})"

# 2. GeoNames 로딩 (힌트용)
def load_gazetteer_for_hint():
    print(f"{CYAN}🌍 유저 거주지 추정을 위해 GeoNames 로딩 중...{END}")
    if not os.path.exists(GEONAMES_FILE): return None
    try:
        cols = ['lat', 'lon', 'admin1 code', 'country code']
        df = pd.read_csv(GEONAMES_FILE, sep='\t', names=['geonameid', 'name', 'asciiname', 'alternatenames', 'lat', 'lon', 'feature class', 'feature code', 'country code', 'cc2', 'admin1 code', 'admin2 code', 'admin3 code', 'admin4 code', 'population', 'elevation', 'dem', 'timezone', 'modification date'], 
                         usecols=cols, header=None, quoting=csv.QUOTE_NONE, encoding='utf-8')
        us_df = df[df['country code'] == 'US'].copy()
        return us_df[['lat', 'lon', 'admin1 code']].to_dict('records')
    except: return None

gazetteer_db = load_gazetteer_for_hint()

def get_user_state_hint(user_loc):
    if not gazetteer_db: return "??"
    u_lat, u_lon = user_loc
    min_dist = float('inf')
    best_state = "??"
    # 속도를 위해 1000개씩 건너뛰며 샘플링 검색하거나, 그냥 다 검색 (여기선 간단히 전체 검색)
    for loc in gazetteer_db:
        if abs(loc['lat'] - u_lat) > 2.0 or abs(loc['lon'] - u_lon) > 2.0: continue
        dist = (loc['lat'] - u_lat)**2 + (loc['lon'] - u_lon)**2
        if dist < min_dist:
            min_dist = dist
            best_state = loc['admin1 code']
    return best_state

# 3. 데이터 로드
INPUT_FILE = None
for f in INPUT_FILES:
    if os.path.exists(f):
        INPUT_FILE = f
        break

if not INPUT_FILE:
    print(f"{RED}❌ 파일 없음.{END}")
    sys.exit()

with open(INPUT_FILE, 'rb') as f:
    all_data = pickle.load(f)

labeled_data = []
if os.path.exists(OUTPUT_FILE):
    with open(OUTPUT_FILE, 'rb') as f:
        labeled_data = pickle.load(f)

processed_ids = {item['user_id'] for item in labeled_data}

print(f"\n{BOLD}🚀 [라벨링 도구 V5] 한글 번역 & 구글맵 & 힌트 풀옵션!{END}")
print(f"📝 총 대상: {len(all_data)}명 / 현재 완료: {len(labeled_data)}명\n")

# 4. 메인 루프
try:
    for item in all_data:
        user_id = item['user_id']
        if user_id in processed_ids: continue
        toponyms = item['toponyms']
        if not toponyms: continue
        
        user_true_loc = item['user_true_loc']
        user_state_hint = get_user_state_hint(user_true_loc)
        user_map_url = f"https://www.google.com/maps?q={user_true_loc[0]},{user_true_loc[1]}"
        
        # 원본 텍스트 정리
        tweet_text = toponyms[0].get('orig_text', "")
        clean_text = tweet_text.replace("RT @", "\n🔄 RT @").strip()
        
        # 🤖 [번역 수행]
        print(f"{CYAN}🤖 번역 중... (GPT-4o-mini){END}")
        translated_text = translate_to_korean(tweet_text) # 원본 통째로 번역
        
        # 화면 출력
        print("\n" + "="*80)
        print(f"👤 ID: {user_id} | 📍 거주지: {BOLD}{user_state_hint}{END} | 🔗 {BLUE}{user_map_url}{END}")
        print("-" * 80)
        
        # 영문 출력
        print(f"{BOLD}[ 🇺🇸 원문 ]{END}")
        # 타겟 단어 강조
        display_eng = clean_text
        for topo in toponyms:
            display_eng = display_eng.replace(topo['word'], f"{YELLOW}{BOLD}**{topo['word']}**{END}")
        print(display_eng)
        
        print("-" * 40)
        
        # 한글 출력
        print(f"{BOLD}[ 🇰🇷 번역 ]{END}")
        print(f"{GREEN}{translated_text}{END}")
        print("=" * 80)
        
        new_toponyms = []
        skip_user = False
        
        for topo in toponyms:
            word = topo['word']
            candidates = topo['candidates']
            
            if len(candidates) < 2:
                topo['human_label_idx'] = 0
                new_toponyms.append(topo)
                continue
            
            print(f"\n🎯 타겟: {YELLOW}{BOLD}** {word} **{END}")
            print(f"👇 어디일까요? (힌트: {CYAN}{user_state_hint}{END})")
            
            for idx, cand in enumerate(candidates):
                star = "⭐" if idx == 0 else "  "
                state_match = ""
                if str(cand['state']) == str(user_state_hint):
                    state_match = f"{GREEN}👈 (거주지 일치){END}"
                
                dist_str = ""
                if 'dist_from_user' in cand:
                    dist_str = f"- {cand['dist_from_user']:.1f}km"
                
                cand_map_url = f"https://www.google.com/maps?q={cand['lat']},{cand['lon']}"
                print(f"   [{idx}] {star} {BOLD}{cand['name']}{END} ({cand['state']}, US) {dist_str} {state_match}")
                print(f"       🔗 {BLUE}{cand_map_url}{END}")
            
            print(f"   [s] Skip")
            print(f"   [q] 종료")
            
            while True:
                choice = input("👉 선택: ").strip().lower()
                if choice == 'q': raise KeyboardInterrupt
                if choice == 's':
                    print(f"{RED}⏩ Skip.{END}")
                    break
                if choice.isdigit() and 0 <= int(choice) < len(candidates):
                    topo['human_label_idx'] = int(choice)
                    new_toponyms.append(topo)
                    print(f"{GREEN}✅ 선택됨!{END}")
                    break
                print("❌ 다시 입력")
        
        if new_toponyms:
            item['toponyms'] = new_toponyms
            labeled_data.append(item)
            
        if len(labeled_data) % 5 == 0:
            with open(OUTPUT_FILE, 'wb') as f:
                pickle.dump(labeled_data, f)
                print("💾 저장됨.")

except KeyboardInterrupt:
    print("\n🛑 중단. 저장 중...")

with open(OUTPUT_FILE, 'wb') as f:
    pickle.dump(labeled_data, f)

print(f"\n🎉 총 {len(labeled_data)}개 완료!")