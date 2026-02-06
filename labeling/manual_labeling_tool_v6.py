import pickle
import os
import sys
import pandas as pd
import csv
import re
import json
from haversine import haversine
from openai import OpenAI

# ======================================================
# 🔑 [필수] API 키 입력
# ======================================================
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") 
# ======================================================

client = OpenAI(api_key=OPENAI_API_KEY)

# 설정 및 색상
INPUT_FILES = ["../test_with_ground_truth_llm.pkl", "../test_with_ground_truth.pkl"]
OUTPUT_FILE = "manual_labeled_data.pkl"
DATA_DIR = "../data"
GEONAMES_FILE = os.path.join(DATA_DIR, "cities1000.txt")

YELLOW = '\033[93m'
CYAN = '\033[96m'
BOLD = '\033[1m'
END = '\033[0m'
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
MAGENTA = '\033[95m' # AI 추천용 색상

# 1. AI 통합 분석 함수 (번역 + 추천 + 이유)
def analyze_with_llm(tweet_text, toponym, candidates, user_state_hint):
    """
    GPT-5-mini에게 트윗, 타겟 단어, 후보군, 유저 거주지를 주고
    1) 번역, 2) 정답 추천, 3) 이유를 JSON으로 받아옴.
    """
    # 후보군 정보를 텍스트로 정리
    cand_str = ""
    for idx, cand in enumerate(candidates):
        dist_info = f"{cand.get('dist_from_user', 9999):.1f}km away" if 'dist_from_user' in cand else "distance unknown"
        cand_str += f"[{idx}] {cand['name']} ({cand['state']}, US) - {dist_info}\n"

    prompt = f"""
    You are a geolocation expert assisting a researcher.
    
    [Context]
    - User's Home State: {user_state_hint} (This is a strong hint. Users usually mention places nearby.)
    - Tweet: "{tweet_text}"
    - Target Toponym: "{toponym}"
    
    [Candidates]
    {cand_str}
    
    [Task]
    1. Translate the tweet into natural Korean (informal/slang style).
    2. Analyze which candidate is the most likely 'Ground Truth' based on the user's home location and context.
    3. Select the best index (0, 1, 2...). If none are likely (e.g. person name, noise), return -1.
    4. Explain your reasoning in Korean.
    
    [Output Format]
    Return a JSON object with keys: "translation", "best_index", "reason".
    """
    
    try:
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            response_format={"type": "json_object"},

        )
        result = json.loads(response.choices[0].message.content)
        return result
    except Exception as e:
        return {"translation": f"(오류: {e})", "best_index": -99, "reason": "API 호출 실패"}

# 2. GeoNames 로딩
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
    for loc in gazetteer_db:
        if abs(loc['lat'] - u_lat) > 2.0 or abs(loc['lon'] - u_lon) > 2.0: continue
        dist = (loc['lat'] - u_lat)**2 + (loc['lon'] - u_lon)**2
        if dist < min_dist:
            min_dist = dist
            best_state = loc['admin1 code']
    return best_state

# 3. 파일 로드
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

print(f"\n{BOLD}🚀 [라벨링 도구 V6] AI Co-Pilot 모드 가동!{END}")
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
        
        # 텍스트 정리
        tweet_text = toponyms[0].get('orig_text', "")
        clean_text = tweet_text.replace("RT @", "\n🔄 RT @").strip()
        
        # 화면 기본 정보 출력
        print("\n" + "="*80)
        print(f"👤 ID: {user_id} | 📍 거주지: {BOLD}{user_state_hint}{END} | 🔗 {BLUE}{user_map_url}{END}")
        print("-" * 80)
        
        # 원문 출력
        print(f"{BOLD}[ 🇺🇸 원문 ]{END}")
        display_eng = clean_text
        for topo in toponyms:
            display_eng = display_eng.replace(topo['word'], f"{YELLOW}{BOLD}**{topo['word']}**{END}")
        print(display_eng)
        print("-" * 80)

        new_toponyms = []
        skip_user = False
        
        for topo in toponyms:
            word = topo['word']
            candidates = topo['candidates']
            
            if len(candidates) < 2:
                topo['human_label_idx'] = 0
                new_toponyms.append(topo)
                continue
            
            # =========================================================
            # 🤖 AI 분석 요청 (번역 + 추천 + 이유)
            # =========================================================
            print(f"{CYAN}⏳ AI가 문맥을 분석 중입니다...{END}", end="\r")
            ai_result = analyze_with_llm(tweet_text, word, candidates, user_state_hint)
            
            # 분석 결과 출력 (보라색 박스)
            print(f"{MAGENTA}{BOLD}╔════════════ [ 🤖 AI 분석 리포트 ] ═══════════════════════════════════╗{END}")
            print(f"{MAGENTA}║ 🇰🇷 번역: {END} {ai_result.get('translation', '...')}")
            print(f"{MAGENTA}╠══════════════════════════════════════════════════════════════════════╣{END}")
            
            best_idx = ai_result.get('best_index', -99)
            rec_str = f"[{best_idx}]번 후보" if best_idx >= 0 else "Skip (지명 아님/불확실)"
            
            print(f"{MAGENTA}║ 💡 추천: {BOLD}{rec_str}{END}")
            print(f"{MAGENTA}║ 📝 이유: {END} {ai_result.get('reason', '...')}")
            print(f"{MAGENTA}{BOLD}╚══════════════════════════════════════════════════════════════════════╝{END}")

            print(f"\n🎯 타겟: {YELLOW}{BOLD}** {word} **{END}")
            
            for idx, cand in enumerate(candidates):
                star = "⭐" if idx == 0 else "  "
                ai_pick = f"{MAGENTA}👈 AI Pick{END}" if idx == best_idx else ""
                
                state_match = ""
                if str(cand['state']) == str(user_state_hint):
                    state_match = f"{GREEN}(거주지){END}"
                
                dist_str = ""
                if 'dist_from_user' in cand:
                    dist_str = f"- {cand['dist_from_user']:.1f}km"
                
                cand_map_url = f"https://www.google.com/maps?q={cand['lat']},{cand['lon']}"
                print(f"   [{idx}] {star} {BOLD}{cand['name']}{END} ({cand['state']}) {dist_str} {state_match} {ai_pick}")
                print(f"       🔗 {BLUE}{cand_map_url}{END}")
            
            print(f"   [s] Skip")
            print(f"   [q] 종료")
            
            while True:
                choice = input(f"👉 선택 (AI 추천: {best_idx}): ").strip().lower()
                if choice == 'q': raise KeyboardInterrupt
                if choice == 's':
                    print(f"{RED}⏩ Skip.{END}")
                    break
                # 빈 엔터 치면 AI 추천값 자동 선택 기능 (편의성)
                if choice == '' and best_idx >= 0 and best_idx < len(candidates):
                     choice = str(best_idx)
                
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