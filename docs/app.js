// 브라우저판: 사진 → 품목(SigLIP 2 기준 사진 벡터 검색) → 유통기한(PaddleOCR + expiry.js) → 확인 → localStorage 목록
import { AutoProcessor, SiglipVisionModel, RawImage, env as hfEnv } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";
import Ocr, { env } from "./vendor/paddleocr.bundle.js"; // @gutenye/ocr-browser 1.4.9 묶음(PaddleOCR PP-OCRv4 + onnxruntime-web)
import { findExpiry } from "./expiry.js";

const MODEL = "onnx-community/siglip2-base-patch16-224-ONNX";
const OCR_MODELS = "https://cdn.jsdelivr.net/npm/@gutenye/ocr-models@1.2.2/";
const OCR_SIDE = 2000; // 원본 PoC(fridge.py MAX_SIDE)와 같다: 큰 사진은 줄이되 작은 날짜 글자는 남긴다
const SAMPLES = [["soymilk_t05.jpg", "두유 (날짜 있음)"], ["milk_t01.jpg", "우유 (날짜 안 보임)"], ["sweetpotato_t01.jpg", "고구마 (날짜 없는 품목)"]];
const KEY = "fridge-items";
const DAY = 86400000;
const $ = (id) => document.getElementById(id);

const items = await (await fetch("items.json")).json();
const byName = Object.fromEntries(items.map((i) => [i.name, i]));
$("name").innerHTML = items.map((i) => `<option>${i.name}</option>`).join("");

// ---- 품목 인식 ----
let modelP = null;
hfEnv.backends.onnx.wasm.proxy = true; // SigLIP 계산도 별도 작업 공간에서
const loadModel = () => (modelP ??= (async () => {
  const [proc, model] = await Promise.all([
    AutoProcessor.from_pretrained(MODEL),
    SiglipVisionModel.from_pretrained(MODEL, { dtype: "q8" }),
  ]);
  const embed = async (img) => {
    const { pooler_output: v } = await model(await proc(img));
    const a = v.data, n = Math.hypot(...a);
    return Array.from(a, (x) => x / n);
  };
  const refs = await (await fetch("refs.json")).json();
  const bank = [];
  for (const r of refs) bank.push([r.name, await embed(await RawImage.fromURL(`refs/${r.file}`))]);
  return { embed, bank };
})());

async function recognize(img) {
  const { embed, bank } = await loadModel();
  const q = await embed(img);
  let best = [null, -1];
  for (const [name, v] of bank) {
    const s = q.reduce((acc, x, i) => acc + x * v[i], 0);
    if (s > best[1]) best = [name, s];
  }
  return best;
}

// ---- 유통기한 ----
let ocrP = null;
const loadOcr = () => (ocrP ??= (async () => {
  env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";
  env.wasm.proxy = true; // OCR 모델 계산을 별도 작업 공간(Web Worker)에서 돌려 화면이 덜 멈추게
  return Ocr.create({ models: {
    detectionPath: `${OCR_MODELS}ch_PP-OCRv4_det_infer.onnx`,
    recognitionPath: `${OCR_MODELS}ch_PP-OCRv4_rec_infer.onnx`,
    dictionaryPath: `${OCR_MODELS}ppocr_keys_v1.txt`,
  } });
})());

function rotated(bitmap, deg, zoom = 1) {
  const scale = Math.min(zoom, OCR_SIDE / Math.max(bitmap.width, bitmap.height));
  const w = Math.round(bitmap.width * scale), h = Math.round(bitmap.height * scale);
  const c = document.createElement("canvas");
  [c.width, c.height] = deg % 180 ? [h, w] : [w, h];
  const g = c.getContext("2d");
  g.translate(c.width / 2, c.height / 2);
  g.rotate((deg * Math.PI) / 180);
  g.drawImage(bitmap, -w / 2, -h / 2, w, h);
  return c;
}

async function readExpiry(bitmap) {
  const ocr = await loadOcr();
  const today = new Date(Date.UTC(...ymdNow()));
  let last = "";
  // 원본의 rotate 방식처럼 돌려 가며 날짜가 나올 때까지.
  // ponytail: 작은 사진은 1.5배로 한 번 더 읽는다(가벼운 모바일 모델이 연도 앞자리를 자주 놓쳐서). 최악 8번 읽기.
  for (const deg of [0, 270, 90, 180]) {
    for (const zoom of [1, 1.5]) {
      const c = rotated(bitmap, deg, zoom);
      const px = c.getContext("2d").getImageData(0, 0, c.width, c.height);
      const { texts } = await ocr.detect({ data: px.data, width: c.width, height: c.height });
      const [d, ev] = findExpiry(texts.map((t) => [t.text.trim(), t.mean]), today);
      if (d) return [d, ev];
      last = ev || last;
    }
  }
  return [null, last];
}

// ---- 화면 ----
function ymdNow() { const t = new Date(); return [t.getFullYear(), t.getMonth(), t.getDate()]; }
const iso = (d) => d.toISOString().slice(0, 10);
const addDays = (n) => iso(new Date(Date.UTC(...ymdNow()) + n * DAY));
// 받침 유무로 조사 고르기: josa("고구마", "은", "는") → "고구마는"
const josa = (w, a, b) => w + ((w.charCodeAt(w.length - 1) - 0xac00) % 28 ? a : b);
const secSince = (t0) => ((performance.now() - t0) / 1000).toFixed(1);
const el = (tag, cls, text) => Object.assign(document.createElement(tag), { className: cls || "", textContent: text ?? "" });

function status(text, busy = false) {
  $("status").replaceChildren(...(busy ? [el("span", "spinner")] : []), el("span", "", text));
}
function facts(pairs) {
  $("facts").replaceChildren(...pairs.map(([k, v]) => {
    const f = el("span", "fact", `${k} `);
    f.append(el("b", "", v));
    return f;
  }));
}

async function handle(blob) {
  $("form").classList.remove("show");
  $("result").classList.add("show");
  $("preview").src = URL.createObjectURL(blob);
  facts([]);
  try {
    const bitmap = await createImageBitmap(blob); // 폰 사진의 회전(EXIF)을 반영한다
    status(modelP ? "품목 인식 중…" : "품목 인식 중… 처음엔 모델을 내려받느라 30초쯤 걸립니다", true);
    let t0 = performance.now();
    const [name, score] = await recognize(await RawImage.fromCanvas(rotated(bitmap, 0)));
    const shown = [["품목", name], ["점수", score.toFixed(2)], ["인식", `${secSince(t0)}초`]];
    $("name").value = name;
    let date = "", msg;
    if (byName[name].dated) {
      facts(shown);
      status("유통기한 읽는 중… 사진을 돌려 가며 날짜를 찾습니다", true);
      t0 = performance.now();
      const [d, ev] = await readExpiry(bitmap);
      date = d ? iso(d) : "";
      shown.push(["날짜 읽기", `${secSince(t0)}초`]);
      if (ev) shown.push(["읽은 글자", ev]);
      msg = d ? `${name}, 유통기한은 ${date}입니다.` : `${name} 같습니다. 날짜를 못 읽었으니 직접 넣어 주세요.`;
    } else {
      const it = byName[name];
      date = addDays(it.default_days);
      shown.push(["기본 보관", `${it.default_days}일`]);
      msg = `${josa(name, "은", "는")} 날짜가 안 적혀 있어 ${it.storage} 기준 ${it.default_days}일로 채웠습니다.`;
    }
    $("date").value = date;
    facts(shown);
    status(msg);
  } catch (e) {
    console.error(e);
    status(`분석하지 못했습니다(${e.message}). 품목과 날짜를 직접 골라 등록할 수 있습니다.`);
  }
  $("form").classList.add("show");
}

const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
const save = (rows) => { try { localStorage.setItem(KEY, JSON.stringify(rows)); } catch { status("이 브라우저는 목록을 저장할 수 없습니다."); } };

function render() {
  const today = Date.UTC(...ymdNow());
  const rows = load().map((r) => ({ ...r, left: Math.round((Date.parse(r.date) - today) / DAY) }))
    .sort((a, b) => a.left - b.left);
  $("empty").hidden = rows.length > 0;
  $("count").textContent = rows.length ? `${rows.length}개` : "";
  $("list").replaceChildren(...rows.map((r) => {
    const li = el("li", "item");
    const cls = r.left < 0 ? "over" : r.left <= 2 ? "soon" : "";
    li.append(el("span", `badge ${cls}`, r.left < 0 ? `${-r.left}일 지남` : r.left === 0 ? "오늘까지" : `D-${r.left}`));
    const what = el("span", "what");
    what.append(el("b", "", r.name), el("small", "", r.date));
    const btn = el("button", "eat", "먹음");
    btn.onclick = () => { save(load().filter((x) => x.id !== r.id)); render(); };
    li.append(what, btn);
    return li;
  }));
}

$("form").onsubmit = (e) => {
  e.preventDefault();
  save([...load(), { id: Date.now(), name: $("name").value, date: $("date").value }]);
  $("form").classList.remove("show");
  status(`${josa($("name").value, "을", "를")} 등록했습니다. 아래 목록을 보세요.`);
  render();
};
$("file").onchange = (e) => e.target.files[0] && handle(e.target.files[0]);
const drop = $("drop");
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => {
  e.preventDefault();
  drop.classList.remove("over");
  const f = e.dataTransfer.files[0];
  if (f && f.type.startsWith("image/")) handle(f);
};
for (const [f, label] of SAMPLES) {
  const b = el("button");
  b.type = "button";
  b.title = label;
  b.append(Object.assign(document.createElement("img"), { src: `samples/${f}`, alt: `시험 사진: ${label}` }));
  b.onclick = async () => handle(await (await fetch(`samples/${f}`)).blob());
  $("samples").append(b);
}
render();
