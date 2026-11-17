"""검증: data/labels.csv의 정답(사진을 돌리기 전에 적는다)과 각 방법의 결과를 비교해 results/에 저장한다.

labels.csv 열: file, name, expiry(YYYY-MM-DD, 찍혀 있지 않으면 비움)
실행: .venv\\Scripts\\python evaluate.py
"""
import csv
import json
from datetime import date
from pathlib import Path

import fridge

ROOT = Path(__file__).parent
METHODS = ["yolo", "world", "siglip", "world+siglip"]


def main():
    labels = list(csv.DictReader((ROOT / "data" / "labels.csv").open(encoding="utf-8")))
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    rows, summary = [], {}
    for m in METHODS:
        ok = 0
        for lb in labels:
            r = fridge.recognize(ROOT / "data" / "photos" / lb["file"], m)
            hit = r["name"] == lb["name"]
            ok += hit
            rows.append({"file": lb["file"], "method": m, "truth": lb["name"], "pred": r["name"],
                         "score": r["score"], "ok": int(hit), "sec": r["sec"]})
        summary[m] = f"{ok}/{len(labels)}"
    dated = [lb for lb in labels if lb["expiry"]]
    d_ok, d_rows = 0, []
    for lb in labels:
        e = fridge.read_expiry(ROOT / "data" / "photos" / lb["file"])
        truth = date.fromisoformat(lb["expiry"]) if lb["expiry"] else None
        hit = e["date"] == truth
        d_ok += hit and truth is not None
        d_rows.append({"file": lb["file"], "truth": truth, "pred": e["date"], "ok": int(hit),
                       "evidence": e["evidence"], "sec": e["sec"], "ocr": " | ".join(e["lines"])[:200]})
    summary["expiry(날짜 찍힌 사진)"] = f"{d_ok}/{len(dated)}"
    summary["no-date 사진에서 엉뚱한 날짜를 만든 수"] = sum(1 for r in d_rows if r["truth"] is None and r["pred"])
    for name, data in (("recognition.csv", rows), ("expiry.csv", d_rows)):
        with (out / name).open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
