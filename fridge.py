"""사진 한 장 → 품목 인식 + 유통기한 읽기. 앱(app.py)과 검증(evaluate.py)이 같이 쓴다.

품목 인식 방법 (모델 비교용):
  yolo    : YOLO26n 기본(COCO 80종). 냉장고 음식 대부분이 목록에 없어 기준선 역할.
  world   : YOLO-World. 품목 이름을 글로 주면 그 물체를 찾는다(open-vocabulary 탐지).
  siglip  : SigLIP 2. 사진(또는 탐지한 영역)이 어느 품목 설명과 맞는지 점수를 매긴다.
  world+siglip : YOLO-World로 위치를 찾고, 잘라 낸 영역의 이름은 SigLIP 2가 정한다.
  ref     : 내가 찍은 기준 사진들을 SigLIP 2 벡터로 저장해 두고, 새 사진과 가장 가까운 품목을 고른다(벡터 검색).
  world+ref : YOLO-World로 잘라 낸 영역을 기준 사진 벡터와 비교한다.
evaluate.py가 방법별 정답률을 재서 가장 좋은 방법을 results/best_method.txt에 적고, 앱은 그것을 기본값으로 쓴다.
유통기한: PaddleOCR(한국어)로 글자를 읽고 expiry.find_expiry 규칙으로 날짜를 고른다.
"""
import json
import os
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).parent
# 이 PC는 사용자 홈의 .cache에 쓸 수 없어서, 모델·캐시를 프로젝트 안 .home/에 둔다(.gitignore 처리).
_HOME = ROOT / ".home"
_HOME.mkdir(exist_ok=True)
for _k in ("USERPROFILE", "HOME"):
    os.environ[_k] = str(_HOME)
# PaddleOCR은 Windows에서 한글이 든 경로의 모델 파일을 못 읽는다 → 영문 경로(LOCALAPPDATA)에 둔다.
os.environ.setdefault("PADDLE_PDX_CACHE_HOME",
                      str(Path(os.environ.get("LOCALAPPDATA", str(_HOME))) / "fridge-keeper" / "paddlex"))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

import torch  # noqa: E402,F401  Windows에서는 paddle보다 torch를 먼저 불러야 DLL 충돌이 안 난다
from PIL import Image  # noqa: E402

from expiry import find_expiry  # noqa: E402

ITEMS = json.loads((ROOT / "items.json").read_text(encoding="utf-8"))["items"]
NAMES = [i["name"] for i in ITEMS]
# COCO 클래스 중 품목과 이어 볼 수 있는 것만 (나머지 품목은 COCO에 없다)
COCO_MAP = {"bottle": "우유", "broccoli": "시금치"}
METHODS = ["yolo", "world", "siglip", "world+siglip", "ref", "world+ref"]
REFS = ROOT / "data" / "refs"  # 기준 사진: data/refs/<품목 이름>/*.jpg (검증 사진과 다른 사진)


def best_method():
    """evaluate.py가 고른 가장 좋은 방법. 아직 검증 전이면 world+siglip."""
    f = ROOT / "results" / "best_method.txt"
    return f.read_text(encoding="utf-8").strip() if f.exists() else "world+siglip"


@lru_cache(None)
def _yolo():
    from ultralytics import YOLO
    return YOLO(str(ROOT / "models" / "yolo26n.pt"))


@lru_cache(None)
def _world():
    from ultralytics import YOLOWorld
    m = YOLOWorld(str(ROOT / "models" / "yolov8s-worldv2.pt"))
    m.set_classes([i["en"] for i in ITEMS])
    return m


@lru_cache(None)
def _siglip():
    from transformers import AutoModel, AutoProcessor
    name = "google/siglip2-base-patch16-224"
    return AutoModel.from_pretrained(name).eval(), AutoProcessor.from_pretrained(name)


@lru_cache(None)
def _ocr():
    from paddleocr import PaddleOCR
    return PaddleOCR(lang="korean", use_doc_orientation_classify=False,
                     use_doc_unwarping=False, use_textline_orientation=False,
                     enable_mkldnn=False)  # paddle 3.3 CPU의 oneDNN 경로 버그(ConvertPirAttribute...) 회피


def _best_box(result):
    """가장 큰 박스 하나(사진 한 장 = 품목 한 개 전제). 반환: (클래스 이름, conf, xyxy) 또는 None."""
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None
    areas = (boxes.xyxy[:, 2] - boxes.xyxy[:, 0]) * (boxes.xyxy[:, 3] - boxes.xyxy[:, 1])
    k = int(areas.argmax())
    return result.names[int(boxes.cls[k])], float(boxes.conf[k]), [float(v) for v in boxes.xyxy[k]]


def _siglip_scores(img):
    import torch
    model, proc = _siglip()
    texts = [f"{i['ko']} 사진" for i in ITEMS]
    inp = proc(text=texts, images=img, padding="max_length", max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.sigmoid(model(**inp).logits_per_image)[0]
    return {n: float(p) for n, p in zip(NAMES, probs)}


def _embed(img):
    """SigLIP 2 이미지 벡터(길이 1로 정규화)."""
    import torch
    model, proc = _siglip()
    with torch.no_grad():
        v = model.get_image_features(**proc(images=img, return_tensors="pt"))
    v = v.pooler_output if hasattr(v, "pooler_output") else v
    return torch.nn.functional.normalize(v, dim=-1)[0]


@lru_cache(None)
def _ref_bank():
    """기준 사진들을 벡터로 바꿔 둔다. 반환: [(품목, 벡터)]."""
    bank = []
    for n in NAMES:
        for p in sorted((REFS / n).glob("*")) if (REFS / n).exists() else []:
            if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                bank.append((n, _embed(Image.open(p).convert("RGB"))))
    return bank


def _ref_scores(img):
    """새 사진 벡터와 품목별 기준 사진 벡터의 코사인 유사도 최댓값."""
    bank = _ref_bank()
    if not bank:
        raise RuntimeError("data/refs/<품목>/ 에 기준 사진이 없습니다")
    q = _embed(img)
    scores = {}
    for n, v in bank:
        scores[n] = max(scores.get(n, -1.0), float(q @ v))
    return scores


def recognize(path, method=None, conf=0.10, sig_threshold=0.01, ref_threshold=0.0):
    method = method or best_method()
    """반환: dict(name, score, box, method, sec). name이 None이면 '모름'(수동 입력으로 넘긴다)."""
    t0 = time.perf_counter()
    img = Image.open(path).convert("RGB")
    name, score, box = None, 0.0, None
    if method == "yolo":
        hit = _best_box(_yolo().predict(img, conf=conf, verbose=False)[0])
        if hit:
            name, score, box = COCO_MAP.get(hit[0]), hit[1], hit[2]
    elif method == "world":
        hit = _best_box(_world().predict(img, conf=conf, verbose=False)[0])
        if hit:
            en2name = {i["en"]: i["name"] for i in ITEMS}
            name, score, box = en2name[hit[0]], hit[1], hit[2]
    elif method in ("siglip", "world+siglip"):
        crop = img
        if method == "world+siglip":
            hit = _best_box(_world().predict(img, conf=conf, verbose=False)[0])
            if hit:
                box = hit[2]
                crop = img.crop(tuple(int(v) for v in box))
        scores = _siglip_scores(crop)
        best = max(scores, key=scores.get)
        name, score = (best, scores[best]) if scores[best] >= sig_threshold else (None, scores[best])
    elif method in ("ref", "world+ref"):
        crop = img
        if method == "world+ref":
            hit = _best_box(_world().predict(img, conf=conf, verbose=False)[0])
            if hit:
                box = hit[2]
                crop = img.crop(tuple(int(v) for v in box))
        scores = _ref_scores(crop)
        best = max(scores, key=scores.get)
        name, score = (best, scores[best]) if scores[best] >= ref_threshold else (None, scores[best])
    else:
        raise ValueError(method)
    return {"name": name, "score": round(score, 4), "box": box, "method": method,
            "sec": round(time.perf_counter() - t0, 2)}


def read_expiry(path):
    """반환: dict(date, evidence, lines, sec). date는 datetime.date 또는 None."""
    t0 = time.perf_counter()
    lines = []
    import numpy as np
    arr = np.array(Image.open(path).convert("RGB"))[:, :, ::-1].copy()  # 한글 경로 대비: 파일 대신 BGR 배열로 넘긴다
    for r in _ocr().predict(arr):
        lines += list(zip(r["rec_texts"], [float(s) for s in r["rec_scores"]]))
    d, ev = find_expiry(lines)
    return {"date": d, "evidence": ev, "lines": [t for t, _ in lines], "sec": round(time.perf_counter() - t0, 2)}


if __name__ == "__main__":
    import sys
    p = sys.argv[1]
    m = sys.argv[2] if len(sys.argv) > 2 else None
    print(recognize(p, m))
    print(read_expiry(p))
