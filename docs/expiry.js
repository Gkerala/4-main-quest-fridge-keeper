// expiry.py의 날짜 고르기 규칙을 그대로 옮긴 것 (브라우저 버전). 확인: node expiry_check.mjs
const PATTERNS = [
  [/(20\d{2})\s*[./\-년]\s*(\d{1,2})\s*[./\-월]\s*(\d{1,2})/g, "ymd4"],
  [/(?<!\d)(\d{2})\s*[./\-]\s*(\d{1,2})\s*[./\-]\s*(\d{1,2})(?!\d)/g, "ymd2"],
  [/(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)/g, "compact"],
  [/(?<!\d)(\d{1,2})\s*[./월]\s*(\d{1,2})(?!\d)/g, "md"],
];
const GOOD = ["소비기한", "유통기한", "까지", "EXP", "exp", "BEST"];
const BAD = ["제조", "MFG", "mfg", "포장일", "생산"];
const MIN_SCORE = 0.5;
const TIME = /^\s*[./]?\s*\d{1,2}:\d{2}/;
const DAY = 86400000;

const ymd = (y, m, d) => {
  const t = new Date(Date.UTC(y, m - 1, d));
  return t.getUTCFullYear() === y && t.getUTCMonth() === m - 1 && t.getUTCDate() === d ? t : null;
};

function toDate(kind, g, today) {
  const [a, b, c] = g.map(Number);
  if (kind === "ymd4" || kind === "compact") return ymd(a, b, c);
  if (kind === "ymd2") return ymd(2000 + a, b, c);
  const d = ymd(today.getUTCFullYear(), a, b); // 연도 없음: 반년 넘게 지났으면 내년으로 본다
  if (!d) return null;
  return (today - d) / DAY <= 180 ? d : ymd(today.getUTCFullYear() + 1, a, b);
}

// lines: [[text, score 0~1]], today: UTC 자정 Date. 반환: [Date 또는 null, 근거 글자]
export function findExpiry(lines, today) {
  const lo = ymd(today.getUTCFullYear() - 1, today.getUTCMonth() + 1, 1);
  const hi = ymd(today.getUTCFullYear() + 3, 12, 31);
  const cands = [];
  lines.forEach(([text, score], i) => {
    const ctx = lines.slice(Math.max(0, i - 1), i + 2).map((l) => l[0]).join(" ");
    const goodCtx = GOOD.some((k) => ctx.includes(k));
    for (const [pat, kind] of PATTERNS) {
      for (const m of text.matchAll(pat)) {
        const d = toDate(kind, m.slice(1), today);
        if (!d) continue;
        if (kind === "md" && !(goodCtx || TIME.test(text.slice(m.index + m[0].length)))) continue;
        if (d < lo || d > hi) continue;
        if (score < MIN_SCORE) continue;
        let w = (goodCtx ? 2 : 0) - (BAD.some((k) => text.includes(k)) ? 3 : 0);
        w += kind !== "md" ? 0.5 : 0;
        cands.push([w, d, text]);
      }
      if (cands.length && cands[cands.length - 1][2] === text) break;
    }
  });
  if (!cands.length) return [null, ""];
  const [w, d, text] = cands.reduce((a, b) => (b[0] > a[0] || (b[0] === a[0] && b[1] > a[1]) ? b : a));
  return w >= 0 ? [d, text] : [null, text];
}
