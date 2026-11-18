// docs/expiry.js가 expiry.py와 같은 답을 내는지 확인 (expiry.py의 시험을 그대로 옮김). 사용: node expiry_check.mjs
import assert from "node:assert/strict";
import { findExpiry } from "./docs/expiry.js";

const D = (y, m, d) => new Date(Date.UTC(y, m - 1, d)).toISOString().slice(0, 10);
const f = (lines, t = [2026, 11, 17]) => {
  const [d] = findExpiry(lines, new Date(Date.UTC(t[0], t[1] - 1, t[2])));
  return d && d.toISOString().slice(0, 10);
};

assert.equal(f([["소비기한 2026.11.30까지", 0.9]]), D(2026, 11, 30));
assert.equal(f([["제조일자 2026.11.10", 0.9], ["소비기한 2026.11.24", 0.9]]), D(2026, 11, 24));
assert.equal(f([["26.12.03", 0.8]]), D(2026, 12, 3));
assert.equal(f([["20261205", 0.8]]), D(2026, 12, 5));
assert.equal(f([["11.20 까지", 0.8]]), D(2026, 11, 20));
assert.equal(f([["국산 시금치", 0.9]]), null);
assert.equal(f([["1병(20g)당9.7 kcal", 0.9]]), null);
assert.equal(f([["11.26.01:33 까지", 0.9]]), D(2026, 11, 26));
assert.equal(f([["11.26.01:33 0F3", 0.9], ["11.14.01:33", 0.9]]), D(2026, 11, 26));
assert.equal(f([["11.14 까지", 0.9]]), D(2026, 11, 14));
assert.equal(f([["09.07 까지", 0.9]], [2027, 6, 1]), D(2027, 9, 7));
assert.equal(f([["02.03 까지", 0.9]]), D(2027, 2, 3));
assert.equal(f([["11.26.01:33 WF3", 0.7], ["11.14.01:33", 0.95]]), D(2026, 11, 26));
assert.equal(f([["2027.07.13D", 0.9]]), D(2027, 7, 13)); // 두유 시험 사진의 실제 OCR 글자
console.log("expiry.js ok");
