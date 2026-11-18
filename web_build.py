"""GitHub Pages(docs/)용 기준 사진 축소본과 목록(refs.json), 체험용 시험 사진을 만든다. 사용: python web_build.py"""
import csv, json, shutil
from pathlib import Path
from PIL import Image, ImageOps

ROOT = Path(__file__).parent
OUT = ROOT / "docs"
SIDE = 448  # SigLIP 입력은 224라 이 정도면 충분, 용량만 줄인다


def shrink(src, dst, side=SIDE):
    img = ImageOps.exif_transpose(Image.open(src)).convert("RGB")
    img.thumbnail((side, side))
    dst.parent.mkdir(parents=True, exist_ok=True)
    img.save(dst, quality=85)


refs = []
for r in csv.DictReader((ROOT / "data/labels.csv").open(encoding="utf-8")):
    shrink(ROOT / "data/photos" / r["file"], OUT / "refs" / r["file"])
    refs.append({"name": r["name"], "file": r["file"]})
for d in sorted((ROOT / "data/refs").iterdir()):
    for p in sorted(d.glob("*.jp*g")) if d.is_dir() else []:
        f = f"{d.name}_{p.name}"
        shrink(p, OUT / "refs" / f)
        refs.append({"name": d.name, "file": f})
(OUT / "refs.json").write_text(json.dumps(refs, ensure_ascii=False, indent=1), encoding="utf-8")

# 체험용 시험 사진(기준 사진과 다른 장). OCR이 글자를 읽어야 해서 크게 둔다.
for f in ["soymilk_t05.jpg", "milk_t01.jpg", "spinach_t01.jpg", "sweetpotato_t01.jpg"]:
    if (ROOT / "data/test" / f).exists():
        shrink(ROOT / "data/test" / f, OUT / "samples" / f, side=2000)
items = json.loads((ROOT / "items.json").read_text(encoding="utf-8"))["items"]
(OUT / "items.json").write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
print(len(refs), "refs")
