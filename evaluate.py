"""검증: 정답 파일(사진을 돌리기 전에 적는다)과 각 방법의 결과를 비교해 results/에 저장한다.

정답 파일 열: file, name, expiry(YYYY-MM-DD, 사진에 날짜가 안 보이면 비움)
  data/labels.csv      : 처음 찍은 사진(data/photos). 기준 사진 방식(ref)의 기준이 된다.
  data/test/labels.csv : 다른 조건에서 다시 찍은 시험 사진(data/test). 있으면 품목 인식은 이것으로 판정한다.

품목 인식
  - 시험 사진이 있으면: 기준 = data/photos 전체, 시험 = data/test  (공정한 판정, 가장 좋은 방법을 여기서 고름)
  - 같은 날 사진끼리(leave-one-out): 기준 = 자기 자신을 뺀 data/photos  (같은 촬영 조건이라 낙관적인 참고값)
유통기한: 읽기 방식(fridge.EXPIRY_MODES)마다, 날짜가 보이는 사진의 정답률과 날짜 없는 사진에서 지어낸 날짜 수.
실행: .venv\\Scripts\\python evaluate.py [--skip-expiry | --expiry-only 방식,방식]  (--expiry-only: 지정한 유통기한 방식만 다시 재서 덮어씀)
"""
import csv
import json
import sys
from datetime import date
from pathlib import Path

import fridge

ROOT = Path(__file__).parent
OUT = ROOT / "results"


def load(folder):
    f = ROOT / "data" / folder / "labels.csv" if folder else ROOT / "data" / "labels.csv"
    if not f.exists():
        return []
    img_dir = ROOT / "data" / (folder or "photos")
    return [dict(r, path=img_dir / r["file"]) for r in csv.DictReader(f.open(encoding="utf-8"))]


def recog(labels, tag):
    rows, acc, sec = [], {}, {}
    for m in fridge.METHODS:
        ok = 0
        for lb in labels:
            r = fridge.recognize(lb["path"], m)
            ok += r["name"] == lb["name"]
            rows.append({"set": tag, "file": lb["file"], "method": m, "truth": lb["name"], "pred": r["name"],
                         "score": r["score"], "ok": int(r["name"] == lb["name"]), "sec": r["sec"]})
        acc[m] = ok / len(labels)
        sec[m] = sum(x["sec"] for x in rows if x["method"] == m) / len(labels)
    return rows, acc, sec


def expiry(labels, modes=None):
    rows, summ = [], {}
    dated = [lb for lb in labels if lb["expiry"]]
    for mode in modes or fridge.EXPIRY_MODES:
        ok = fake = 0
        t = 0.0
        for lb in labels:
            e = fridge.read_expiry(lb["path"], mode)
            truth = date.fromisoformat(lb["expiry"]) if lb["expiry"] else None
            ok += bool(truth) and e["date"] == truth
            fake += truth is None and e["date"] is not None
            t += e["sec"]
            rows.append({"mode": mode, "file": lb["file"], "truth": truth, "pred": e["date"],
                         "ok": int(e["date"] == truth), "angle": e["angle"], "evidence": e["evidence"],
                         "sec": e["sec"], "ocr": " | ".join(e["lines"])[:200]})
        summ[mode] = {"정답": f"{ok}/{len(dated)} ({ok / len(dated):.0%})", "지어낸 날짜": fake,
                      "평균 초": round(t / len(labels), 1)}
    return rows, summ


def save_csv(name, rows):
    with (OUT / name).open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def expiry_only(modes):
    """지정한 방식만 다시 재서 summary.json과 expiry.csv의 해당 방식 줄만 바꾼다."""
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    erows, esumm = expiry(load(None), modes)
    summary.setdefault("유통기한", {}).update(esumm)
    old = []
    if (OUT / "expiry.csv").exists():
        old = [r for r in csv.DictReader((OUT / "expiry.csv").open(encoding="utf-8-sig")) if r["mode"] not in modes]
    save_csv("expiry.csv", old + erows)
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary["유통기한"], ensure_ascii=False, indent=1))


def main():
    OUT.mkdir(exist_ok=True)
    if "--expiry-only" in sys.argv:
        return expiry_only(sys.argv[sys.argv.index("--expiry-only") + 1].split(","))
    photos, test = load(None), load("test")
    summary = {}
    rows, acc_loo, sec_loo = recog(photos, "같은날(LOO)")
    summary["품목 인식 · 같은 날 사진(낙관적)"] = {m: f"{a:.0%}" for m, a in acc_loo.items()}
    acc, sec = acc_loo, sec_loo
    if test:
        trows, acc, sec = recog(test, "시험사진")
        rows += trows
        summary["품목 인식 · 시험 사진(판정용)"] = {m: f"{a:.0%}" for m, a in acc.items()}
    best = max(acc, key=lambda m: (acc[m], -sec[m]))  # 정답률이 같으면 빠른 쪽
    (OUT / "best_method.txt").write_text(best, encoding="utf-8")
    summary["가장 좋은 방법"] = best + ("" if test else " (시험 사진 없음: 낙관적 값으로 고름)")
    summary["평균 초"] = {m: round(s, 2) for m, s in sec.items()}
    save_csv("recognition.csv", rows)
    if test:  # 시험 사진의 품목별 정답률 (어느 품목에서 약한지)
        per = {}
        for m in fridge.METHODS:
            for n in fridge.NAMES:
                rs = [r for r in rows if r["set"] == "시험사진" and r["method"] == m and r["truth"] == n]
                if rs:
                    per.setdefault(m, {})[n] = f"{sum(r['ok'] for r in rs)}/{len(rs)}"
        summary["시험 사진 품목별"] = per
    if "--skip-expiry" not in sys.argv:
        erows, esumm = expiry(photos + [t for t in test if "expiry" in t])
        summary["유통기한"] = esumm
        save_csv("expiry.csv", erows)
    elif (OUT / "summary.json").exists():  # 유통기한은 지난 결과를 그대로 둔다
        old = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
        if "유통기한" in old:
            summary["유통기한"] = old["유통기한"]
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
