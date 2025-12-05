# AI Styler SLM 패키지 운영 및 학습 매뉴얼 (v1.2)

이 문서는 AI Styler 시스템의 SLM(Small Language Model) 패키지를 고객이 직접 활용하여 **데이터 증강·모델 재학습·API 서버 구동·운영 연동**을 수행할 수 있도록 제공되는 전용 기술 문서입니다.

---

## 전체 코드 다운로드 (필수)

이 저장소에는 용량 문제로 학습 코드 및 체크포인트가 포함되어 있지 않습니다.
**아래 구글 드라이브에서 전체 파일을 다운로드한 후 진행하세요.**

### 다운로드 링크

**[구글 드라이브에서 전체 코드 다운로드](https://drive.google.com/file/d/1YqmNQHaKOBkQo3Yr9q5R6yULBpVZNX-D/view?usp=sharing)**

### 다운로드 후 설치 방법

```bash
# 1. 다운로드한 파일 압축 해제
unzip koreatimes_training_pipeline.zip

# 2. 압축 해제된 폴더로 이동
cd koreatimes_training_pipeline

# 3. 이후 아래 매뉴얼을 따라 진행
```

### 포함된 파일

| 항목 | 설명 |
|------|------|
| 학습 스크립트 | `train_detection_lora.py`, `train_correction_lora.py` 등 |
| 데이터 증강 스크립트 | `bulk_augmentation.py`, `violation_injector.py` 등 |
| API 서버 | `test_Inference/api_server.py` |
| 사전 학습 체크포인트 | `checkpoint_2c_466/`, `detection_checkpoint_3300/`, `correction_checkpoint_3200/` |
| 스타일 가이드 | `style_guides.json` (83개 규칙) |

---

## 1. SLM 패키지 개요

### 1.1 목적

기사 데이터가 일정량(약 10,000건 이상) 누적된 이후, 코리아타임스가 자체적으로 모델을 재학습하여 품질을 높이고 유지보수할 수 있도록 지원합니다.

### 1.2 핵심 변경 사항 (GPT 모델 대비)

| 항목 | GPT 모델 | SLM 모델 |
|------|----------|----------|
| 아키텍처 | 단일 모델 | **Detection(감지) + Correction(교정)** 2모델 구조 |
| 베이스 모델 | OpenAI GPT-4/5 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| 운영 방식 | 외부 API 호출 | 내부 GPU 서버에서 독자 운영 |
| 비용 | API 호출당 과금 | 초기 GPU 비용 후 무제한 사용 |

---

## 2. 하드웨어 및 소프트웨어 요구사양

### 2.1 GPU 서버 사양

| 항목 | 최소 사양 | 권장 사양 |
|------|----------|----------|
| GPU | NVIDIA RTX 3090 (24GB) | NVIDIA A100 (40GB+) |
| RAM | 16GB | 32GB |
| 디스크 | 20GB 여유 공간 | 50GB SSD |
| OS | Ubuntu 20.04 LTS | Ubuntu 22.04 LTS |
| CUDA | 11.8 이상 | 12.x |

### 2.2 소프트웨어 환경

| 항목 | 버전 | 비고 |
|------|------|------|
| Python | 3.10 (필수) | Conda 환경 권장 |
| PyTorch | 2.0+ | CUDA 지원 필수 |
| Unsloth | 최신 | GPU 학습/추론 가속 |
| Transformers | 4.35+ | HuggingFace |
| FastAPI | 0.104+ | API 서버용 |

### 2.3 외부 API 키 (데이터 증강 시에만 필요)

- **OpenAI GPT-5 API Key**: 학습 데이터 생성(Augmentation) 단계에서만 사용
- 학습 완료 후 추론 시에는 불필요

---

## 3. 패키지 파일 구성

```text
koreatimes_training_pipeline/
├── README.md                              # 이 파일 (운영 매뉴얼)
├── setup_environment.sh                   # 환경 자동 설정 스크립트
├── requirements.txt                       # Python 의존성 목록
│
├── style_guides.json                      # 83개 스타일 규칙 정의
│                                          # (Title: H01-H11, Body: A01-A39, Caption: C01-C33)
│
│── [데이터 증강]
├── violation_injector.py                  # 위반 주입 핵심 엔진 (3-Expert 구조)
├── bulk_augmentation.py                   # 대량 데이터 증강 스크립트
├── augment_multiple.py                    # 다중 기사 증강 (대안)
├── convert_augmentation_to_training.py    # 증강 결과 → 학습 데이터 변환
│
│── [모델 학습]
├── train_detection_lora.py                # Detection 모델 LoRA 학습
├── train_correction_lora.py               # Correction 모델 LoRA 학습
├── evaluate_v2_lora.py                    # 학습된 모델 성능 평가
│
└── test_Inference/                        # [운영] 추론 및 API 서버
    ├── api_server.py                      # REST API 서버 (FastAPI) ← CMS 연동
    ├── inference_2c.py                    # CLI 기반 테스트 도구
    ├── inference_simple.py                # 간단한 추론 테스트
    │
    ├── checkpoint_2c_466/                 # 사전 학습된 통합 체크포인트
    ├── detection_checkpoint_3300/         # Detection 전용 체크포인트
    └── correction_checkpoint_3200/        # Correction 전용 체크포인트
```

---

## 4. 환경 설정

### 4.1 자동 설치 (권장)

```bash
cd koreatimes_training_pipeline
bash setup_environment.sh
```

스크립트가 자동으로 수행하는 작업:
- Conda 설치 확인
- `korea_times` 환경 생성 (Python 3.10)
- 필수 패키지 설치 (OpenAI, PyTorch, Transformers 등)
- GPU 감지 및 안내

### 4.2 수동 설치

```bash
# 1. Conda 환경 생성
conda create -n korea_times python=3.10 -y

# 2. 환경 활성화
conda activate korea_times

# 3. 기본 패키지 설치
pip install -r requirements.txt

# 4. GPU 환경: Unsloth 설치 (Linux + CUDA 필수)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

### 4.3 설치 확인

```bash
# GPU 확인
nvidia-smi

# Python 환경 확인
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

---

## 5. 데이터 준비 및 증강 (Data Augmentation)

SLM 학습을 위해서는 **'정상 기사'**와 **'오류가 포함된 기사'**의 쌍(Pair)이 필요합니다.

### 5.1 입력 데이터 형식

기사는 **반드시** 아래 태그 형식을 준수해야 합니다:

```text
[TITLE]
Samsung announces new smartphone
[/TITLE]

[BODY]
Samsung Electronics Chairman Lee Jae-yong unveiled the latest Galaxy model on Monday. The device costs 1,200,000 won and features advanced AI capabilities.
[/BODY]

[CAPTION]
Samsung Electronics Chairman Lee Jae-yong presents the new Galaxy phone at the launch event in Seoul on Monday. Yonhap.
[/CAPTION]
```

### 5.2 OpenAI API 키 설정

데이터 증강에는 GPT-5 API가 필요합니다.

```bash
export OPENAI_API_KEY='sk-your-api-key-here'
```

### 5.3 위반 주입 (Augmentation) 실행

정상 기사에 스타일 가이드 위반을 자동 주입합니다.

```python
from bulk_augmentation import augment_articles_bulk
import asyncio

# 기사 리스트 준비
articles = [
    '[TITLE]\nSamsung announces...[/TITLE]\n[BODY]...[/BODY]\n[CAPTION]...[/CAPTION]',
    # 더 많은 기사...
]

# 증강 실행
asyncio.run(augment_articles_bulk(
    articles=articles,
    difficulties=['basic', 'intermediate', 'advanced'],
    model='gpt-5',
    reasoning_effort='low',
    text_verbosity='low'
))
```

**난이도 설정:**
| 난이도 | 주입되는 위반 개수 | 용도 |
|--------|-------------------|------|
| `basic` | 1-2개 | 쉬운 케이스 학습 |
| `intermediate` | 3-5개 | 중간 난이도 |
| `advanced` | 6-10개 | 복잡한 케이스 |

**출력 파일:** `bulk_augmentation_results_YYYYMMDD_HHMMSS.json`

### 5.4 학습 데이터 변환

증강 결과를 Detection/Correction 학습용 JSONL로 변환합니다.

```bash
python3 convert_augmentation_to_training.py \
    --input bulk_augmentation_results_*.json \
    --train-ratio 0.8 \
    --seed 42
```

**출력 구조:**
```text
detection_dataset/
├── detection_train.jsonl    # Detection 학습용
└── detection_val.jsonl      # Detection 검증용

correction_dataset/
├── correction_train.jsonl   # Correction 학습용
└── correction_val.jsonl     # Correction 검증용
```

---

## 6. SLM 모델 재학습 (Training)

Detection과 Correction 두 개의 모델을 **각각** 학습해야 합니다.

### 6.1 학습 파라미터

| 파라미터 | 값 | 설명 |
|----------|-----|------|
| 베이스 모델 | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | 4-bit 양자화 |
| LoRA Rank | 32 | 학습 가능 파라미터 수 |
| LoRA Alpha | 32 | 스케일링 팩터 |
| Dropout | 0.05 | 과적합 방지 |
| Max Seq Length | 4096 | 최대 토큰 길이 |

### 6.2 Detection (감지) 모델 학습

기사 내 오류의 **위치와 종류**를 찾아내는 모델입니다.

```bash
python3 train_detection_lora.py \
    --train-data detection_dataset/detection_train.jsonl \
    --val-data detection_dataset/detection_val.jsonl \
    --output-dir detection_lora_v1 \
    --max-steps 5000 \
    --batch-size 4
```

### 6.3 Correction (교정) 모델 학습

감지된 오류를 **수정**하는 모델입니다.

```bash
python3 train_correction_lora.py \
    --train-data correction_dataset/correction_train.jsonl \
    --val-data correction_dataset/correction_val.jsonl \
    --output-dir correction_lora_v1 \
    --max-steps 5000 \
    --batch-size 4
```

### 6.4 학습 결과

- **출력:** LoRA 어댑터 (약 166MB 각 모델)
- **소요 시간:** 100 기사 기준 약 1시간 (모델당)
- **저장 위치:** `--output-dir`로 지정한 폴더

### 6.5 모델 평가 (선택)

```bash
# Detection 모델 평가
python3 evaluate_v2_lora.py \
    --model-path detection_lora_v1 \
    --test-file detection_dataset/detection_val.jsonl \
    --task detection

# Correction 모델 평가
python3 evaluate_v2_lora.py \
    --model-path correction_lora_v1 \
    --test-file correction_dataset/correction_val.jsonl \
    --task correction
```

**평가 지표:**
- Detection: Rule-level F1, Precision, Recall
- Correction: Exact Match, Text Similarity

---

## 7. SLM 모델 운영 적용 (Deployment)

### 7.1 API 서버 구동

학습된 체크포인트를 로드하여 REST API 서버를 실행합니다.

```bash
cd koreatimes_training_pipeline/test_Inference

# 기본 실행 (포트 8081)
python3 api_server.py --checkpoint checkpoint_2c_466

# 포트 지정
python3 api_server.py --port 8081 --checkpoint checkpoint_2c_466

# 새로 학습한 모델 사용 시
python3 api_server.py --port 8081 --checkpoint ../detection_lora_v1
```

**백그라운드 실행 (운영 환경):**
```bash
nohup python3 api_server.py --port 8081 --checkpoint checkpoint_2c_466 > api_server.log 2>&1 &
```

### 7.2 서버 상태 확인

| 확인 방법 | 명령어/URL |
|-----------|------------|
| Swagger UI | `http://<GPU_SERVER_IP>:8081/docs` |
| Health Check | `curl http://<GPU_SERVER_IP>:8081/health` |
| 모델 정보 | `curl http://<GPU_SERVER_IP>:8081/model/info` |

### 7.3 API 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| `POST` | `/generate` | 기사 교정 (메인) |
| `GET` | `/health` | 서버 상태 확인 |
| `GET` | `/model/info` | 모델 정보 조회 |

### 7.4 API 호출 예시

**요청:**
```bash
curl -X POST "http://localhost:8081/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "[TITLE]\nSamsung announces new smartphone\n[/TITLE]\n\n[BODY]\nSamsung Electronics Chairman Lee Jae-yong unveiled the latest Galaxy model on Monday.\n[/BODY]\n\n[CAPTION]\nSamsung Electronics Chairman Lee presents the new Galaxy phone. Yonhap.\n[/CAPTION]",
    "temperature": 0.7,
    "top_p": 0.8,
    "top_k": 20,
    "max_tokens": 2048
  }'
```

**응답:**
```json
{
  "corrected_text": "[TITLE]...[/TITLE]\n\n[BODY]...[/BODY]\n\n[CAPTION]...[/CAPTION]",
  "violations": [
    {
      "rule_id": "A01",
      "component_type": "body",
      "violation_type": "formatting",
      "original_text": "1,200,000 won",
      "violated_text": "1.2 million won",
      "description": "Large numbers should use million/billion format"
    }
  ],
  "processing_time_ms": 1234.5
}
```

### 7.5 AI Styler 백엔드 연동

CMS 백엔드가 SLM API 서버를 바라보도록 설정합니다.

**`backend/.env` 파일 수정:**
```ini
# SLM 서버 사용 활성화
USE_AI_SERVER=true

# SLM API 서버 주소 (GPU 서버 IP 및 포트)
AI_SERVER_URL=http://<GPU_SERVER_IP>:8081/generate

# OpenAI 사용 비활성화
USE_OPENAI=false
```

**백엔드 재시작:**
```bash
docker-compose restart backend
```

### 7.6 사용 가능한 체크포인트

| 체크포인트 | 용도 | 크기 |
|------------|------|------|
| `checkpoint_2c_466/` | Detection + Correction 통합 | 333MB |
| `detection_checkpoint_3300/` | Detection 전용 | 167MB |
| `correction_checkpoint_3200/` | Correction 전용 | 167MB |

---

## 8. CLI 테스트 (선택)

API 서버 없이 로컬에서 테스트할 수 있습니다.

### 8.1 인터랙티브 모드

```bash
cd test_Inference

python3 inference_2c.py \
    --checkpoint checkpoint_2c_466 \
    --interactive
```

### 8.2 파일 입력 모드

```bash
python3 inference_2c.py \
    --checkpoint checkpoint_2c_466 \
    --input article.txt \
    --output result.json
```

---

## 9. 운영 주의사항

### 9.1 서비스 안정성

| 항목 | 권장 사항 |
|------|----------|
| 프로세스 관리 | `systemd` 또는 `supervisor`로 서비스 등록 |
| 로그 관리 | `logrotate` 설정으로 로그 파일 관리 |
| 모니터링 | `/health` 엔드포인트 주기적 체크 |
| GPU 메모리 | 단일 요청 처리 권장 (동시 요청 시 OOM 주의) |

### 9.2 systemd 서비스 등록 예시

`/etc/systemd/system/slm-api.service`:
```ini
[Unit]
Description=Korea Times SLM API Server
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/path/to/koreatimes_training_pipeline/test_Inference
ExecStart=/path/to/conda/envs/korea_times/bin/python api_server.py --port 8081 --checkpoint checkpoint_2c_466
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable slm-api
sudo systemctl start slm-api
```

---

## 10. 장애 대응 및 롤백

### 10.1 GPT 모델로 롤백 (긴급 복구)

SLM 서버 장애 시 즉시 GPT 모델로 전환할 수 있습니다.

**`backend/.env` 수정:**
```ini
USE_AI_SERVER=false
USE_OPENAI=true
OPENAI_API_KEY=sk-your-openai-key
```

**백엔드 재시작:**
```bash
docker-compose restart backend
```

### 10.2 일반적인 문제 해결

| 증상 | 원인 | 해결 방법 |
|------|------|----------|
| `ModuleNotFoundError: unsloth` | GPU 환경 아님 | CUDA 설치 후 unsloth 재설치 |
| `CUDA out of memory` | GPU 메모리 부족 | `--batch-size` 감소 또는 다른 요청 종료 |
| API 서버 응답 없음 | 프로세스 종료됨 | `systemctl restart slm-api` |
| 교정 품질 저하 | 학습 데이터 부족 | 증강 데이터 추가 후 재학습 |

### 10.3 로그 확인

```bash
# API 서버 로그
tail -f api_server.log

# systemd 서비스 로그
journalctl -u slm-api -f
```

---

## 11. 전체 워크플로우 요약

```
┌─────────────────────────────────────────────────────────────────┐
│  0. 전체 코드 다운로드 (구글 드라이브)                            │
│     https://drive.google.com/file/d/1YqmNQHaKOBkQo3Yr9q5R6yULBpVZNX-D │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  1. 환경 설정                                                    │
│     bash setup_environment.sh                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  2. 데이터 증강 (GPT-5 API 사용)                                 │
│     bulk_augmentation.py → 정상 기사에 위반 주입                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  3. 학습 데이터 변환                                             │
│     convert_augmentation_to_training.py                         │
│     → detection_dataset/, correction_dataset/ 생성              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  4. 모델 학습 (GPU 필수)                                         │
│     train_detection_lora.py  → Detection 모델                   │
│     train_correction_lora.py → Correction 모델                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  5. API 서버 구동                                                │
│     python3 api_server.py --port 8081 --checkpoint <모델경로>   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  6. CMS 연동                                                     │
│     backend/.env에서 AI_SERVER_URL 설정                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 부록 A. 스타일 규칙 체계

| 컴포넌트 | 규칙 ID | 개수 |
|----------|---------|------|
| Title (제목) | H01-H11 | 11개 |
| Body (본문) | A01-A39 | 39개 |
| Caption (캡션) | C01-C33 | 33개 |
| **합계** | | **83개** |

상세 규칙은 `style_guides.json` 파일 참조.

---

## 부록 B. 버전 이력

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| v1.0 | 2025-11-13 | 초기 작성 |
| v1.1 | 2025-12-01 | API 서버 추가, 명령어 구체화 |
| v1.2 | 2025-12-05 | 베이스 모델 ID 수정, LoRA 파라미터 명시, 파일 구조 보완, systemd 예시 추가, 구글 드라이브 다운로드 안내 추가 |

---

**문서 버전:** v1.2
**최종 업데이트:** 2025-12-05
