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

echo "🌙 [Mega Grid Search V4] 총 $TOTAL_EXP 개의 실험 시작..."

for h in "${HIDDEN_LIST[@]}"
do
    for d in "${DROPOUT_LIST[@]}"
    do
        CURRENT_EXP=$((CURRENT_EXP + 1))
        LOG_FILE="$RESULT_DIR/log_h${h}_d${d}.txt"
        
        echo "--------------------------------------------------"
        echo "🧪 [Progress: $CURRENT_EXP / $TOTAL_EXP] Running: Hidden=$h, Dropout=$d"
        
        # [중요] 헷갈리지 않게 기존 model.pkl 삭제
        rm -f model.pkl

        # 1. 학습 실행
        THEANO_FLAGS='device=cuda1,floatX=float32,force_device=True' python gcnmain.py -hid $h -dropout $d > "$LOG_FILE" 2>&1
        
        # 2. 모델 파일 찾기 (이름이 model.pkl 이든 model-*.pkl 이든 다 찾음)
        if [ -f "model.pkl" ]; then
             echo "📦 표준 이름(model.pkl) 발견! 테스트 진행..."
             # 이름이 이미 model.pkl 이므로 복사할 필요 없음
        else
             # model.pkl이 없으면 model-*.pkl 중 가장 최신 것 찾기
             LATEST_MODEL=$(ls -t model-*.pkl 2>/dev/null | head -1)
             if [ -n "$LATEST_MODEL" ]; then
                 echo "📦 가변 이름($LATEST_MODEL) 발견 -> model.pkl 로 복사"
                 cp "$LATEST_MODEL" model.pkl
             else
                 echo "⚠️ 학습 실패: 모델 파일이 아예 생성되지 않았습니다."
                 ACCURACY="Train_Fail"
             fi
        fi

        # 3. 테스트 실행 (model.pkl이 존재할 때만)
        if [ -f "model.pkl" ]; then
             TEST_OUTPUT=$(THEANO_FLAGS='device=cuda1,floatX=float32,force_device=True' python gcn_test_only.py -hid $h -dropout $d)
             ACCURACY=$(echo "$TEST_OUTPUT" | grep "FINAL TEST ACCURACY" | awk '{print $5}')
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
