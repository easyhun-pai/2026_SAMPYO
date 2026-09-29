# 2026_SAMPYO

![pipeline](docs/pipeline.png)

## To-do

### 환경
- [x] `.venv` 생성 — torch 2.6+cu124, transformers, opencv
- [x] Hugging Face 가입 → DINOv3 라이선스 동의 → 토큰 발급 → `hf auth login`
- [ ] **DINOv3 게이트 승인 대기 중** ([Gated Repos](https://huggingface.co/settings/gated-repos)) → 승인되면 재추출:
  `python scripts/embed_crops.py --model facebook/dinov3-vitb16-pretrain-lvd1689m`

### 데이터
- [x] 특징 추출 (`scripts/embed_crops.py`, 현재 DINOv2-base) → `__DATA__/features/embeddings.npy`
- [x] 근접 중복 묶기 (`scripts/dedup_crops.py --thr 0.80`) → 19,473장 → **4,742 묶음**
- [x] 라벨링 툴 (`scripts/label_ppe.py`, `라벨링툴.bat`) — 묶음 대표 1장 라벨 → 묶음 전체에 전파
- [ ] 라벨링 기준 확정
  - 스마트조끼 O: 위 주황 · 아래 검정 (앞/뒤/옆 모두)
  - 스마트조끼 X: 일반 안전조끼(연두·노랑), 하네스, 미착용
  - 판단불가: 너무 작거나 가려짐 → 학습 제외
- [ ] **스마트조끼 라벨링** — 큰 묶음부터 500개(= 이미지 72% 커버) 목표
- [ ] 안전모 라벨링 — 미착용은 카메라3에만 존재 (40묶음)
- [ ] 1차 모델 예측 → 애매한 것만 재검수
- [ ] train / val / test 분리: 시간 구간·영상 단위 (랜덤 분리 금지)

### 모델
- [ ] `PPEClassifier`: DINOv3 백본 + 안전모 헤드 + 스마트조끼 헤드 → 4클래스 조합
- [ ] 라벨 없는 속성은 손실에서 제외 (스마트조끼만 먼저 학습 가능)
- [ ] 1단계: 백본 고정 + 헤드 학습 / 2단계: 마지막 2~4 블록 미세조정
- [ ] 평가: 속성별 X(위반) 재현율, 4×4 혼동행렬

### 현장 (복귀 후)
- [ ] 안전모 미착용 추가 촬영: 카메라1·2, 여러 명, 스마트조끼 착용 + 안전모 미착용 포함
- [ ] 트래킹 ID별 다중 프레임 투표
- [ ] 스마트조끼 판정은 `stair_check.py` 트럭 위(DECK) 작업자에만 적용
- [ ] 서비스 화면/문서에 "Built with DINOv3" 표기 (라이선스)
