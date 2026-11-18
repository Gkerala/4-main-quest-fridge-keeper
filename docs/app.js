// 브라우저판: 사진 → 품목(SigLIP 2 기준 사진 벡터 검색) → 유통기한(Tesseract.js + expiry.js) → 확인 → localStorage 목록
import { AutoProcessor, SiglipVisionModel, RawImage } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1";
import Ocr, { env } from "./vendor/paddleocr.bundle.js"; // @gutenye/ocr-browser 1.4.9 묶음(PaddleOCR PP-OCRv4 + onnxruntime-web)
import { findExpiry } from "./expiry.js";

const MODEL = "onnx-community/siglip2-base-patch16-224-ONNX";
const OCR_MODELS = "https://cdn.jsdelivr.net/npm/@gutenye/ocr-models@1.2.2/";
const OCR_SIDE = 2000; // 원본 PoC(fridge.py MAX_SIDE)와 같다: 큰 사진은 줄이되 작은 날짜 글자는 남긴다
const SAMPLES = ["soymilk_t05.jpg", "milk_t01.jpg", "sweetpotato_t01.jpg"];
const KEY = "fridge-items";
const DAY = 86400000;
const $ = (id) => document.getElementById(id);

const items = await (await fetch("items.json")).json();
const byName = Object.fromEntries(items.map((i) => [i.name, i]));
$("name").innerHTML = items.map((i) => `<option>${i.name}</option>`).join("");

// ---- 품목 인식 ----
let modelP = null;
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
const status = (t) => ($("status").textContent = t);

async function handle(blob) {
  $("form").style.display = "none";
  $("preview").src = URL.createObjectURL(blob);
  $("preview").style.display = "block";
  try {
    const bitmap = await createImageBitmap(blob); // 폰 사진의 회전(EXIF)을 반영한다
    const small = rotated(bitmap, 0);
    status("품목 인식 중… (처음엔 모델 내려받기 때문에 오래 걸립니다)");
    let t0 = performance.now();
    const [name, score] = await recognize(await RawImage.fromCanvas(small));
    const recSec = ((performance.now() - t0) / 1000).toFixed(1);
    $("name").value = name;
    let date = "", note = "";
    if (byName[name].dated) {
      status(`인식: ${name} (점수 ${score.toFixed(3)}, ${recSec}초) · 유통기한 읽는 중…`);
      t0 = performance.now();
      const [d, ev] = await readExpiry(bitmap);
      const sec = ((performance.now() - t0) / 1000).toFixed(1);
      date = d ? iso(d) : "";
      note = d ? `유통기한 ${date} (${sec}초, 근거: ${ev})` : `날짜를 못 읽었습니다(${sec}초) → 직접 입력하세요`;
    } else {
      date = addDays(byName[name].default_days);
      note = `날짜 없는 품목 → 기본 ${byName[name].default_days}일 (${byName[name].storage})`;
    }
    $("date").value = date;
    status(`인식: ${name} (점수 ${score.toFixed(3)}, ${recSec}초) · ${note} · 틀리면 고친 뒤 등록하세요.`);
    $("form").style.display = "block";
  } catch (e) {
    console.error(e);
    status(`실패: ${e.message}. 품목과 날짜를 직접 골라 등록할 수 있습니다.`);
    $("form").style.display = "block";
  }
}

const load = () => { try { return JSON.parse(localStorage.getItem(KEY)) || []; } catch { return []; } };
const save = (rows) => { try { localStorage.setItem(KEY, JSON.stringify(rows)); } catch { status("이 브라우저는 목록을 저장할 수 없습니다."); } };

function render() {
  const today = Date.UTC(...ymdNow());
  const rows = load().map((r) => ({ ...r, left: Math.round((Date.parse(r.date) - today) / DAY) }))
    .sort((a, b) => a.left - b.left);
  $("empty").style.display = rows.length ? "none" : "block";
  $("list").innerHTML = "";
  for (const r of rows) {
    const li = document.createElement("li");
    const cls = r.left < 0 ? "over" : r.left <= 2 ? "soon" : "";
    const tag = r.left < 0 ? `지남 ${-r.left}일` : `D-${r.left}`;
    li.innerHTML = `<span><b>${r.name}</b> · ${r.date} · <span class="${cls}">${tag}</span></span>`;
    const btn = Object.assign(document.createElement("button"), { className: "ghost", textContent: "먹음" });
    btn.onclick = () => { save(load().filter((x) => x.id !== r.id)); render(); };
    li.append(btn);
    $("list").append(li);
  }
}

$("add").onclick = () => {
  if (!$("date").value) return status("유통기한을 넣어 주세요.");
  save([...load(), { id: Date.now(), name: $("name").value, date: $("date").value }]);
  $("form").style.display = "none";
  status(`${$("name").value} 등록했습니다.`);
  render();
};
$("file").onchange = (e) => e.target.files[0] && handle(e.target.files[0]);
for (const f of SAMPLES) {
  const img = Object.assign(document.createElement("img"), { src: `samples/${f}`, alt: f, title: f });
  img.onclick = async () => handle(await (await fetch(img.src)).blob());
  $("samples").append(img);
}
render();
