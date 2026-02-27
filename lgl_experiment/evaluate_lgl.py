#!/usr/bin/env python3
"""
LGL Benchmark Evaluation Pipeline
==================================
GCN 기반 지명 모호성 해소 연구 - 외부 벤치마크 검증

파이프라인:
  1. LGL XML 파싱 → 지명 + 정답 좌표 추출
  2. Gazetteer(cities1000) 후보 검색 → 동명 도시 목록
  3. 채점 방법별 후보 선택:
     - Population-only: 가장 인구 많은 후보
     - Gravity Model: Pop / Dist^β (anchor 필요)
  4. Acc@161km 평가

사용법:
  python evaluate_lgl.py
"""

import xml.etree.ElementTree as ET
import math
import json
from collections import defaultdict
import geonamescache

# ============================================================
# 핵심 함수들 (기존 파이프라인에서 그대로 가져옴)
# ============================================================

def haversine(lat1, lon1, lat2, lon2):
    """두 좌표 간 거리 (km) - 기존 코드와 동일"""
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


def gravity_score(pop, dist_km, beta=1.5):
    """Gravity Model 점수 - 기존 코드와 동일
    Score = Population / Distance^β
    """
    if dist_km < 1.0:
        dist_km = 1.0  # 0km 방지
    return pop / (dist_km ** beta)


# ============================================================
# STEP 1: Gazetteer 구축 (이름 → 후보 도시 목록)
# ============================================================

def build_gazetteer():
    """geonamescache에서 이름별 후보 도시 인덱스 구축
    기존 코드의 gazetteer.py와 동일한 역할
    """
    gc = geonamescache.GeonamesCache()
    cities = gc.get_cities()
    
    # 이름별 후보 목록 (name → [{geonameid, lat, lon, pop, country, ...}, ...])
    name_index = defaultdict(list)
    
    for gid, city in cities.items():
        if city.get('population', 0) < 1000:
            continue
            
        entry = {
            'geonameid': str(city['geonameid']),
            'name': city['name'],
            'lat': city['latitude'],
            'lon': city['longitude'],
            'pop': city['population'],
            'country': city['countrycode'],
            'admin1': city.get('admin1code', ''),
        }
        
        # 정식 이름으로 인덱싱
        name_index[city['name'].lower()].append(entry)
        
        # alternate names로도 인덱싱
        for alt in city.get('alternatenames', []):
            if alt and len(alt) > 1:
                name_index[alt.lower()].append(entry)
    
    return name_index


# ============================================================
# STEP 2: LGL XML 파싱
# ============================================================

def parse_lgl(xml_path='lgl.xml'):
    """LGL XML에서 지명 데이터 추출
    
    Returns:
        toponyms: [{phrase, gt_lat, gt_lon, gt_geonameid, docid, feedid, domain, ...}]
        articles: {docid: {feedid, domain, all_toponym_coords: [(lat,lon), ...]}}
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    toponyms = []
    articles = {}
    
    for art in root.findall('.//article'):
        docid = art.get('docid')
        feedid = art.find('feedid').text
        domain = art.find('domain').text
        text = art.find('text').text or ''
        
        art_coords = []
        
        for topo in art.findall('.//toponym'):
            gaztag = topo.find('gaztag')
            if gaztag is None:
                continue
            
            phrase = topo.find('phrase').text
            lat_el = gaztag.find('lat')
            lon_el = gaztag.find('lon')
            
            if lat_el is None or lon_el is None:
                continue
            
            gt_lat = float(lat_el.text)
            gt_lon = float(lon_el.text)
            geonameid = gaztag.get('geonameid', '')
            
            fclass = gaztag.find('fclass')
            fclass = fclass.text if fclass is not None else ''
            
            admin1_el = gaztag.find('admin1')
            admin1 = admin1_el.text if admin1_el is not None else ''
            
            country_el = gaztag.find('country')
            country = country_el.text if country_el is not None else ''
            
            art_coords.append((gt_lat, gt_lon))
            
            toponyms.append({
                'phrase': phrase,
                'gt_lat': gt_lat,
                'gt_lon': gt_lon,
                'gt_geonameid': geonameid,
                'gt_fclass': fclass,
                'gt_admin1': admin1,
                'gt_country': country,
                'docid': docid,
                'feedid': feedid,
                'domain': domain,
            })
        
        articles[docid] = {
            'feedid': feedid,
            'domain': domain,
            'coords': art_coords,
        }
    
    return toponyms, articles


# ============================================================
# STEP 3: Anchor 계산 - 신문사별 centroid
# ============================================================

def compute_newspaper_centroids(toponyms):
    """각 신문사(feedid)의 지명 centroid를 anchor로 사용
    
    현재 파이프라인에서 GCN이 예측하는 유저 좌표와 동일한 역할
    """
    feed_coords = defaultdict(list)
    
    for t in toponyms:
        feed_coords[t['feedid']].append((t['gt_lat'], t['gt_lon']))
    
    centroids = {}
    for fid, coords in feed_coords.items():
        avg_lat = sum(c[0] for c in coords) / len(coords)
        avg_lon = sum(c[1] for c in coords) / len(coords)
        centroids[fid] = (avg_lat, avg_lon)
    
    return centroids


def compute_article_centroids(toponyms):
    """각 기사 내 다른 지명들의 centroid를 anchor로 사용
    (leave-one-out: 현재 지명을 제외한 나머지 지명들의 중심)
    """
    # docid별 모든 좌표 수집
    doc_coords = defaultdict(list)
    for t in toponyms:
        doc_coords[t['docid']].append((t['gt_lat'], t['gt_lon']))
    
    return doc_coords


# ============================================================
# STEP 4: 평가 실행
# ============================================================

def evaluate(toponyms, gazetteer, method='population', 
             newspaper_centroids=None, beta=1.5):
    """
    지명 해소 평가
    
    method:
        'population': 가장 인구 많은 후보 선택 (기존 Population baseline)
        'gravity_newspaper': Gravity Model + 신문사 centroid anchor
        'gravity_article': Gravity Model + 기사 내 centroid anchor
    """
    results = {
        'total': 0,
        'found_in_gazetteer': 0,
        'correct_161km': 0,
        'errors_km': [],
        'details': [],
    }
    
    for t in toponyms:
        phrase = t['phrase']
        gt_lat, gt_lon = t['gt_lat'], t['gt_lon']
        results['total'] += 1
        
        # Gazetteer 검색 - 기존 gazetteer.py와 동일한 로직
        candidates = gazetteer.get(phrase.lower(), [])
        
        if not candidates:
            # 후보 없음 → 오답 처리 (무한 거리)
            results['errors_km'].append(99999)
            results['details'].append({
                'phrase': phrase,
                'status': 'no_candidates',
                'error_km': 99999,
            })
            continue
        
        results['found_in_gazetteer'] += 1
        
        # ---- 후보 선택 ----
        selected = None
        
        if method == 'population':
            # Population-only: 가장 인구 많은 후보
            selected = max(candidates, key=lambda c: c['pop'])
            
        elif method == 'gravity_newspaper':
            # Gravity Model + 신문사 anchor
            anchor = newspaper_centroids.get(t['feedid'])
            if anchor is None:
                selected = max(candidates, key=lambda c: c['pop'])
            else:
                anchor_lat, anchor_lon = anchor
                best_score = -1
                for c in candidates:
                    dist = haversine(anchor_lat, anchor_lon, c['lat'], c['lon'])
                    score = gravity_score(c['pop'], dist, beta=beta)
                    if score > best_score:
                        best_score = score
                        selected = c
        
        # ---- 정답 비교 ----
        if selected:
            error_km = haversine(gt_lat, gt_lon, selected['lat'], selected['lon'])
            is_correct = error_km <= 161.0
            
            results['errors_km'].append(error_km)
            if is_correct:
                results['correct_161km'] += 1
            
            results['details'].append({
                'phrase': phrase,
                'status': 'correct' if is_correct else 'wrong',
                'error_km': round(error_km, 1),
                'selected': f"{selected['name']} ({selected['country']}, pop={selected['pop']})",
                'gt': f"({gt_lat}, {gt_lon}) {t['gt_country']}",
                'n_candidates': len(candidates),
            })
    
    return results


def print_results(results, method_name):
    """결과 출력"""
    total = results['total']
    found = results['found_in_gazetteer']
    correct = results['correct_161km']
    errors = sorted(results['errors_km'])
    
    acc = correct / total * 100 if total > 0 else 0
    acc_found = correct / found * 100 if found > 0 else 0
    mean_error = sum(errors) / len(errors) if errors else 0
    median_error = errors[len(errors)//2] if errors else 0
    
    # 99999 제외한 실제 에러 통계
    real_errors = [e for e in errors if e < 99999]
    mean_real = sum(real_errors) / len(real_errors) if real_errors else 0
    median_real = real_errors[len(real_errors)//2] if real_errors else 0
    
    print(f"\n{'='*60}")
    print(f"  {method_name}")
    print(f"{'='*60}")
    print(f"  전체 지명 수:           {total}")
    print(f"  Gazetteer 매칭 성공:    {found} ({found/total*100:.1f}%)")
    print(f"  Gazetteer 매칭 실패:    {total - found}")
    print(f"")
    print(f"  ✅ Acc@161km (전체):    {correct}/{total} = {acc:.2f}%")
    print(f"  ✅ Acc@161km (매칭만):  {correct}/{found} = {acc_found:.2f}%")
    print(f"")
    print(f"  Mean Error (매칭만):    {mean_real:.1f} km")
    print(f"  Median Error (매칭만):  {median_real:.1f} km")
    print(f"{'='*60}")
    
    # 오답 샘플 (상위 10개)
    wrong = [d for d in results['details'] if d['status'] == 'wrong']
    if wrong:
        print(f"\n  오답 샘플 (상위 10개):")
        for d in wrong[:10]:
            print(f"    {d['phrase']:20s} → {d['selected']:40s} | err={d['error_km']:>8.1f}km | 후보수={d['n_candidates']}")
    
    return {
        'method': method_name,
        'total': total,
        'found': found,
        'correct': correct,
        'acc_total': round(acc, 2),
        'acc_found': round(acc_found, 2),
        'mean_error': round(mean_real, 1),
        'median_error': round(median_real, 1),
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("🌍 LGL Benchmark Evaluation Pipeline")
    print("=" * 60)
    
    # 1. Gazetteer 구축
    print("\n[1/4] Gazetteer 구축 중 (geonamescache)...")
    gazetteer = build_gazetteer()
    unique_names = len(gazetteer)
    total_entries = sum(len(v) for v in gazetteer.values())
    print(f"  → {unique_names:,} 고유 이름, {total_entries:,} 총 엔트리")
    
    # 2. LGL 파싱
    print("\n[2/4] LGL XML 파싱 중...")
    toponyms, articles = parse_lgl('lgl.xml')
    print(f"  → {len(toponyms)} 지명, {len(articles)} 기사")
    
    # P클래스만 필터 (도시/마을 - cities1000과 매칭 가능)
    p_class_toponyms = [t for t in toponyms if t['gt_fclass'] == 'P']
    all_toponyms = toponyms  # 전체도 저장
    
    print(f"  → P클래스(도시/마을): {len(p_class_toponyms)}")
    print(f"  → A클래스(행정구역): {len([t for t in toponyms if t['gt_fclass'] == 'A'])}")
    
    # 3. Anchor 계산
    print("\n[3/4] Anchor 계산 중...")
    newspaper_centroids = compute_newspaper_centroids(toponyms)
    print(f"  → {len(newspaper_centroids)} 신문사 centroid 계산 완료")
    
    # 4. 평가 실행
    print("\n[4/4] 평가 실행 중...")
    
    summary = []
    
    # === 실험 1: P클래스 - Population-only ===
    r1 = evaluate(p_class_toponyms, gazetteer, method='population')
    s1 = print_results(r1, "P클래스 | Population-only (기존 baseline)")
    summary.append(s1)
    
    # === 실험 2: P클래스 - Gravity + 신문사 anchor (β=1.5) ===
    r2 = evaluate(p_class_toponyms, gazetteer, method='gravity_newspaper',
                  newspaper_centroids=newspaper_centroids, beta=1.5)
    s2 = print_results(r2, "P클래스 | Gravity (β=1.5) + 신문사 centroid anchor")
    summary.append(s2)
    
    # === 실험 3: P클래스 - Gravity + 신문사 anchor (β=1.0) ===
    r3 = evaluate(p_class_toponyms, gazetteer, method='gravity_newspaper',
                  newspaper_centroids=newspaper_centroids, beta=1.0)
    s3 = print_results(r3, "P클래스 | Gravity (β=1.0) + 신문사 centroid anchor")
    summary.append(s3)
    
    # === 실험 4: 전체(P+A) - Population-only ===
    r4 = evaluate(all_toponyms, gazetteer, method='population')
    s4 = print_results(r4, "전체(P+A) | Population-only")
    summary.append(s4)
    
    # === 실험 5: 전체(P+A) - Gravity + 신문사 anchor ===
    r5 = evaluate(all_toponyms, gazetteer, method='gravity_newspaper',
                  newspaper_centroids=newspaper_centroids, beta=1.5)
    s5 = print_results(r5, "전체(P+A) | Gravity (β=1.5) + 신문사 centroid anchor")
    summary.append(s5)
    
    # === 최종 요약 ===
    print("\n" + "=" * 80)
    print("📊 최종 비교 요약")
    print("=" * 80)
    print(f"{'방법':<50s} {'Acc@161km':>10s} {'Mean Err':>10s} {'Median':>10s}")
    print("-" * 80)
    for s in summary:
        print(f"{s['method']:<50s} {s['acc_total']:>9.2f}% {s['mean_error']:>9.1f}km {s['median_error']:>9.1f}km")
    
    print(f"\n💡 참고: 기존 논문 LGL Population baseline = 68.5% (Acc@161km)")
    print(f"   현재 트위터 데이터 결과: Population-only = 88.51%, Full Gravity = 92.34%")
    
    # 결과 JSON 저장
    with open('lgl_results.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: lgl_results.json")
