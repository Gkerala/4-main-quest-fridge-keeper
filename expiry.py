"""OCR 결과 줄들에서 유통기한(소비기한) 날짜를 고른다. 모델 없이 규칙만 쓰므로 단독 시험이 가능하다."""
import re
from datetime import date

# 2026.11.30 / 26.11.30 / 2026-11-30 / 2026/11/30 / 20261130 / 11.30 (연도 없음)
_PATTERNS = [
    (re.compile(r"(20\d{2})\s*[./\-년]\s*(\d{1,2})\s*[./\-월]\s*(\d{1,2})"), "ymd4"),
    (re.compile(r"(?<!\d)(\d{2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{1,2})(?!\d)"), "ymd2"),
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), "compact"),
    (re.compile(r"(?<!\d)(\d{1,2})\s*[./월]\s*(\d{1,2})(?!\d)"), "md"),
]
_GOOD = ("소비기한", "유통기한", "까지", "EXP", "exp", "BEST")
_BAD = ("제조", "MFG", "mfg", "포장일", "생산")


def _to_date(kind, g, today):
    try:
        if kind in ("ymd4", "compact"):
            return date(int(g[0]), int(g[1]), int(g[2]))
        if kind == "ymd2":
            return date(2000 + int(g[0]), int(g[1]), int(g[2]))
        d = date(today.year, int(g[0]), int(g[1]))  # 연도 없음: 이미 지났으면 내년으로 본다
        return d if d >= today else date(today.year + 1, d.month, d.day)
    except ValueError:
        return None


def find_expiry(lines, today=None):
    """lines: [(text, score)]. 반환: (date 또는 None, 근거 문자열)."""
    today = today or date.today()
    cands = []
    for i, (text, score) in enumerate(lines):
        ctx = " ".join(t for t, _ in lines[max(0, i - 1): i + 2])  # 앞뒤 줄도 문맥으로 본다
        for pat, kind in _PATTERNS:
            for m in pat.finditer(text):
                d = _to_date(kind, m.groups(), today)
                if not d:
                    continue
                w = score + (2 if any(k in ctx for k in _GOOD) else 0) - (3 if any(k in text for k in _BAD) else 0)
                w += 0.5 if kind != "md" else 0  # 연도가 있는 쪽을 조금 더 믿는다
                cands.append((w, d, text))
            if cands and cands[-1][2] == text:
                break  # 한 줄에서 긴 형식이 잡히면 짧은 형식으로 다시 잡지 않는다
    if not cands:
        return None, ""
    w, d, text = max(cands, key=lambda c: (c[0], c[1]))  # 점수가 같으면 늦은 날짜(제조일보다 기한일 가능성)
    return (d, text) if w > 0 else (None, text)


if __name__ == "__main__":
    t = date(2026, 11, 17)
    assert find_expiry([("소비기한 2026.11.30까지", 0.9)], t)[0] == date(2026, 11, 30)
    assert find_expiry([("제조일자 2026.11.10", 0.9), ("소비기한 2026.11.24", 0.9)], t)[0] == date(2026, 11, 24)
    assert find_expiry([("26.12.03", 0.8)], t)[0] == date(2026, 12, 3)
    assert find_expiry([("20261205", 0.8)], t)[0] == date(2026, 12, 5)
    assert find_expiry([("11.20 까지", 0.8)], t)[0] == date(2026, 11, 20)
    assert find_expiry([("국산 시금치", 0.9)], t)[0] is None
    print("expiry.py ok")
