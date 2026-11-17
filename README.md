# fridge-keeper: 사진 한 장으로 냉장고 음식과 유통기한 등록하기

> 아이펠 AI에이전트 3기 · 03 모듈 Main Quest 2 (내 도메인에서 AI로 개선 지점 찾아서 PoC 만들기)

- 문제 정의서: [`문제정의서.md`](문제정의서.md)
- 흐름: **사진 → 품목 인식(YOLO 계열 / SigLIP 2) → 유통기한 읽기(PaddleOCR + 날짜 규칙) → 확인·수정 → 등록 → D-day 목록**

## 실행 방법 (Windows, CPU)

```powershell
uv venv --python 3.12 .venv
uv pip install --python .venv\Scripts\python.exe torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
.venv\Scripts\streamlit run app.py          # 앱
.venv\Scripts\python evaluate.py            # 검증 (data/labels.csv 기준)
python expiry.py                            # 날짜 규칙 단독 시험
```

처음 실행할 때 모델을 내려받는다: `models/yolo26n.pt`, `models/yolov8s-worldv2.pt`(ultralytics), `google/siglip2-base-patch16-224`(Hugging Face), PP-OCRv5(PaddleOCR).

## 파일

| 파일 | 역할 |
|---|---|
| `fridge.py` | 품목 인식 4가지 방법 + 유통기한 읽기 |
| `expiry.py` | OCR 글자 줄에서 소비기한 날짜를 고르는 규칙 (제조일자는 제외) |
| `app.py` | Streamlit 앱: 사진 등록 / 직접 등록 / 목록. 등록 시간과 수정 여부를 `logs/timing.csv`에 남김 |
| `evaluate.py` | `data/labels.csv` 정답과 비교해 `results/`에 표 저장 |
| `items.json` | 관리 품목 목록 |

## 모델 선정 (비교 후보)

| 방법 | 무엇 | 왜 후보인가 |
|---|---|---|
| `yolo` | YOLO26n 기본(COCO 80종) | 기준선. 우유팩·시금치·닭가슴살이 COCO에 없다 |
| `world` | YOLO-World v2 | 품목 이름을 글로 주면 찾는다 |
| `siglip` | SigLIP 2 | 사진과 한국어 설명의 일치 점수 |
| `world+siglip` | 위치는 YOLO-World, 이름은 SigLIP 2 | 두 모델의 장점 결합 |

<!-- 검증 결과, 실패 사례, 한계와 다음 단계, 시연 스크린샷은 사진을 모은 뒤 채운다 -->

## 이 PC에서 겪은 설치 문제 (재현용 메모)

- PaddlePaddle은 Python 3.14 바퀴 파일이 없어 3.12로 만들었다.
- paddleocr가 opencv를 여러 개 설치하면 `cv2` import가 깨진다 → `opencv-contrib-python` 하나만 남긴다.
- Windows에서 paddle을 torch보다 먼저 불러오면 torch DLL 오류 → `fridge.py`가 torch를 먼저 불러온다.
- 한글이 든 경로의 Paddle 모델 파일을 못 읽는다 → 모델 캐시를 `%LOCALAPPDATA%\fridge-keeper`에 둔다.
- paddle 3.3 CPU의 oneDNN 경로 오류 → `enable_mkldnn=False`.
