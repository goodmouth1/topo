#!/bin/bash

# 1. 결과 저장 설정
RESULT_DIR="/home/wang/master_thesis/gyulgwa"
mkdir -p "$RESULT_DIR"

REPORT_FILE="$RESULT_DIR/final_summary.txt"
echo "Timestamp | Hidden | Dropout | Test Accuracy | Log File" > "$REPORT_FILE"
echo "==========================================================" >> "$REPORT_FILE"

# 2. 파라미터 목록 (6x6=36개)
HIDDEN_LIST=(100 200 300 400 500 600)
DROPOUT_LIST=(0.1 0.2 0.3 0.4 0.5 0.6)

TOTAL_EXP=${#HIDDEN_LIST[@]}
TOTAL_EXP=$((TOTAL_EXP * ${#DROPOUT_LIST[@]}))
CURRENT_EXP=0

echo "🌙 [Mega Grid Search V5] 총 $TOTAL_EXP 개의 실험 시작..."

for h in "${HIDDEN_LIST[@]}"
do
    for d in "${DROPOUT_LIST[@]}"
    do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        LOG_FILE="$RESULT_DIR/log_h${h}_d${d}.txt"
        
        echo "--------------------------------------------------"
        echo "🧪 [Progress: $CURRENT_EXP / $TOTAL_EXP] Running: Hidden=$h, Dropout=$d"
        
        # 헷갈리지 않게 현재 폴더의 model.pkl 삭제
        rm -f model.pkl

        # 1. 학습 실행 (이제 무조건 저장됨!)
        THEANO_FLAGS='device=cuda1,floatX=float32,force_device=True' python gcnmain.py -hid $h -dropout $d > "$LOG_FILE" 2>&1
        
        # 2. 모델 파일 찾기 (data 폴더 안쪽까지 수색!)
        # ./data/model-*.pkl 중 가장 최신 것 찾기
        LATEST_MODEL=$(ls -t ./data/model-*.pkl 2>/dev/null | head -1)

        if [ -n "$LATEST_MODEL" ]; then
             echo "📦 모델 발견($LATEST_MODEL) -> 현재 폴더로 가져오기"
             cp "$LATEST_MODEL" model.pkl
             
             # 3. 테스트 실행
             TEST_OUTPUT=$(THEANO_FLAGS='device=cuda1,floatX=float32,force_device=True' python gcn_test_only.py -hid $h -dropout $d)
             ACCURACY=$(echo "$TEST_OUTPUT" | grep "FINAL TEST ACCURACY" | awk '{print $5}')
        else
             echo "⚠️ 학습 실패: data 폴더 안에도 모델이 없습니다."
             ACCURACY="Train_Fail"
        fi

        # 값 비어있으면 Error 표시
        if [ -z "$ACCURACY" ]; then
            ACCURACY="Error"
        fi

        # 4. 리포트 기록
        NOW=$(date "+%Y-%m-%d %H:%M")
        echo "$NOW |   $h   |   $d   |   $ACCURACY   | $LOG_FILE" >> "$REPORT_FILE"
        
        echo "✅ 완료! 결과: $ACCURACY"
        
        sleep 3
    done
done

echo "🎉 실험 종료!"
