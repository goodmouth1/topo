import pandas as pd
import glob
import os

# result 폴더 안의 모든 csv 파일 찾기
csv_files = glob.glob('result/*.csv')

if not csv_files:
    print("🚨 'result' 폴더 안에 CSV 파일이 없습니다! 파일명이나 경로를 확인해주세요.")
else:
    print(f"📂 총 {len(csv_files)}개의 결과 파일을 통합 분석합니다...\n")
    print("="*70)
    
    all_errors = []
    
    for file in csv_files:
        model_name = os.path.basename(file)
        df = pd.read_csv(file)
        
        # 4번(Proposed Model)이 틀린 케이스만 추출
        errors = df[df['4. Proposed_Model (결과)'] == 'X'].copy()
        
        # 어느 모델에서 나온 결과인지 꼬리표(출처) 달아주기
        errors.insert(0, 'Source_Model', model_name) 
        all_errors.append(errors)
        
        print(f"📌 [{model_name}] 전체 {len(df)}개 중 오답: {len(errors)}개")
    
    print("="*70)
    
    # 모든 오답 데이터를 하나로 합치기
    combined_errors = pd.concat(all_errors, ignore_index=True)
    
    # 3개 모델이 '공통으로' 틀린 극악의 난이도 지명 찾기
    # (같은 User_ID와 Toponym에서 연속으로 틀린 횟수 카운트)
    error_counts = combined_errors.groupby(['User_ID', 'Toponym', 'Ground_Truth (정답)']).size().reset_index(name='Fail_Count')
    hardest_cases = error_counts[error_counts['Fail_Count'] == len(csv_files)]
    
    print(f"\n🚨 3개 모델이 전부 다 틀린 '극악의 오답(Hardest Cases)'은 총 {len(hardest_cases)}개입니다.")
    if len(hardest_cases) > 0:
        for idx, row in hardest_cases.iterrows():
            print(f"   - 유저: {row['User_ID']} | 헷갈린 지명: {row['Toponym']} | 🎯 진짜 정답: {row['Ground_Truth (정답)']}")
    
    # 터미널에서 오답 미리보기 (최대 5개)
    print("\n🔍 [통합 오답 케이스 미리보기]")
    for idx, row in combined_errors.head(5).iterrows():
        print(f"[{row['Source_Model']}] 유저: {row['User_ID']} | 지명: '{row['Toponym']}'")
        print(f"   🎯 정답: {row['Ground_Truth (정답)']} ❌ AI 선택: {row['4. Proposed_Model (선택)']}")
        print("-" * 70)
    
    # 통합 오답 파일 저장
    out_file = 'result/combined_hard_cases.csv'
    combined_errors.to_csv(out_file, index=False, encoding='utf-8-sig')
    print(f"\n✅ 전체 오답을 하나로 모은 통합 파일이 생성되었습니다: {out_file}")
    print("이 파일을 열어보시면 3개 모델의 약점을 완벽하게 비교하실 수 있습니다!")
