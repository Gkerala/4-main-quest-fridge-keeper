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
MIN_SCORE = 0.5
_BAD = ("제조", "MFG", "mfg", "포장일", "생산")
_TIME = re.compile(r"^\s*[./]?\s*\d{1,2}:\d{2}")  # 'MM.DD HH:MM' (우유처럼 시각까지 찍는 기한 표기)


def _to_date(kind, g, today):
    try:
        if kind in ("ymd4", "compact"):
            return date(int(g[0]), int(g[1]), int(g[2]))
        if kind == "ymd2":
            return date(2000 + int(g[0]), int(g[1]), int(g[2]))
        d = date(today.year, int(g[0]), int(g[1]))  # 연도 없음: 반년 넘게 지났으면 내년으로 본다
        return d if (today - d).days <= 180 else date(today.year + 1, d.month, d.day)  # 며칠 지난 건 기한 지남
    except ValueError:
        return None


def find_expiry(lines, today=None, strict=True):
    """lines: [(text, score)]. 반환: (date 또는 None, 근거 문자열).
    strict=True(개선판): 연도 없는 'MM.DD'는 앞뒤 줄에 까지·소비기한 같은 말이 있을 때만 인정하고,
    오늘 기준 1년 전~3년 뒤 밖의 날짜는 버린다. ('9.7 kcal'을 9월 7일로 읽는 오탐 방지)"""
    today = today or date.today()
    cands = []
    for i, (text, score) in enumerate(lines):
        ctx = " ".join(t for t, _ in lines[max(0, i - 1): i + 2])  # 앞뒤 줄도 문맥으로 본다
        good_ctx = any(k in ctx for k in _GOOD)
        for pat, kind in _PATTERNS:
            for m in pat.finditer(text):
                d = _to_date(kind, m.groups(), today)
                if not d:
                    continue
                if strict and kind == "md" and not (good_ctx or _TIME.match(text[m.end():])):
                    continue
                if strict and not (date(today.year - 1, today.month, 1) <= d <= date(today.year + 3, 12, 31)):
                    continue
                if score < MIN_SCORE:
                    continue  # 흐릿하게 읽힌 줄은 버린다
                # OCR 확신도는 섞지 않는다: 기한·제조 두 줄이 나란히 있으면 확신도 차이로 제조일이 뽑혔다(우유)
                w = (2 if good_ctx else 0) - (3 if any(k in text for k in _BAD) else 0)
                w += 0.5 if kind != "md" else 0  # 연도가 있는 쪽을 조금 더 믿는다
                cands.append((w, d, text))
            if cands and cands[-1][2] == text:
                break  # 한 줄에서 긴 형식이 잡히면 짧은 형식으로 다시 잡지 않는다
    if not cands:
        return None, ""
    w, d, text = max(cands, key=lambda c: (c[0], c[1]))  # 점수가 같으면 늦은 날짜(제조일보다 기한일 가능성)
    return (d, text) if w >= 0 else (None, text)  # 제조일 문맥뿐이면 날짜 없음


if __name__ == "__main__":
    t = date(2026, 11, 17)
    assert find_expiry([("소비기한 2026.11.30까지", 0.9)], t)[0] == date(2026, 11, 30)
    assert find_expiry([("제조일자 2026.11.10", 0.9), ("소비기한 2026.11.24", 0.9)], t)[0] == date(2026, 11, 24)
    assert find_expiry([("26.12.03", 0.8)], t)[0] == date(2026, 12, 3)
    assert find_expiry([("20261205", 0.8)], t)[0] == date(2026, 12, 5)
    assert find_expiry([("11.20 까지", 0.8)], t)[0] == date(2026, 11, 20)
    assert find_expiry([("국산 시금치", 0.9)], t)[0] is None
    assert find_expiry([("1병(20g)당9.7 kcal", 0.9)], t)[0] is None            # 개선판: 오탐 방지
    assert find_expiry([("1병(20g)당9.7 kcal", 0.9)], t, strict=False)[0] is not None  # 처음 버전은 오탐
    assert find_expiry([("11.26.01:33 까지", 0.9)], t)[0] == date(2026, 11, 26)
    # 우유: 'MM.DD HH:MM' 두 줄(기한·제조), 문맥 단어 없음 → 시각 표기로 인정, 늦은 날짜(기한)
    assert find_expiry([("11.26.01:33 0F3", 0.9), ("11.14.01:33", 0.9)], t)[0] == date(2026, 11, 26)
    assert find_expiry([("11.14 까지", 0.9)], t)[0] == date(2026, 11, 14)  # 사흘 지난 날짜는 내년이 아니라 기한 지남
    assert find_expiry([("09.07 까지", 0.9)], date(2027, 6, 1))[0] == date(2027, 9, 7)
    assert find_expiry([("02.03 까지", 0.9)], t)[0] == date(2027, 2, 3)  # 반년 넘게 지남 → 내년
    assert find_expiry([("11.26.01:33 WF3", 0.7), ("11.14.01:33", 0.95)], t)[0] == date(2026, 11, 26)  # 확신도가 낮아도 늦은 날짜
    print("expiry.py ok")
