"""냉장고 앱 PoC (Streamlit). 실행: .venv\\Scripts\\streamlit run app.py

사진 등록: 사진 → 품목 인식 → 유통기한 읽기 → 확인(틀리면 고침) → 등록
수동 등록: 품목과 날짜를 직접 고름
등록할 때마다 걸린 시간과 AI 결과를 고쳤는지를 logs/timing.csv에 남긴다(검증용).
"""
import csv
import sqlite3
import time
from datetime import date, datetime, timedelta
from pathlib import Path

import streamlit as st

import fridge

ROOT = Path(__file__).parent
DB = ROOT / "fridge.db"
LOG = ROOT / "logs" / "timing.csv"
UP = ROOT / "uploads"


def db():
    con = sqlite3.connect(DB)
    con.execute("""create table if not exists items(
        id integer primary key, name text not null, expiry text, source text,
        photo text, created text, eaten integer default 0)""")
    return con


def log(mode, sec, ai_name, ai_date, final_name, final_date):
    LOG.parent.mkdir(exist_ok=True)
    new = not LOG.exists()
    with LOG.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "mode", "sec", "ai_name", "ai_date", "final_name", "final_date", "name_fixed", "date_fixed"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), mode, round(sec, 1), ai_name, ai_date,
                    final_name, final_date, int(ai_name != final_name), int(str(ai_date) != str(final_date))])


def add(name, expiry, source, photo=""):
    with db() as con:
        con.execute("insert into items(name, expiry, source, photo, created) values(?,?,?,?,?)",
                    (name, expiry.isoformat() if expiry else None, source, photo, datetime.now().isoformat(timespec="seconds")))


def default_expiry(name):
    days = next((i["default_days"] for i in fridge.ITEMS if i["name"] == name), None)
    return date.today() + timedelta(days=days) if days else None


st.set_page_config(page_title="냉장고 유통기한", page_icon="🥛")
st.title("내 냉장고 유통기한")
tab_photo, tab_manual, tab_list = st.tabs(["사진으로 등록", "직접 등록", "목록"])

with tab_photo:
    method = st.selectbox("품목 인식 방법", ["world+siglip", "world", "siglip", "yolo"])
    shot = st.file_uploader("음식 사진", type=["jpg", "jpeg", "png"]) or st.camera_input("또는 바로 찍기")
    if shot:
        if st.session_state.get("shot_id") != shot.file_id:  # 새 사진이면 시간 재기 시작
            st.session_state.update(shot_id=shot.file_id, t0=time.perf_counter(), result=None)
        UP.mkdir(exist_ok=True)
        path = UP / f"{datetime.now():%Y%m%d_%H%M%S}_{shot.name if hasattr(shot, 'name') else 'camera.jpg'}"
        if st.session_state["result"] is None:
            path.write_bytes(shot.getvalue())
            with st.spinner("인식 중..."):
                st.session_state["result"] = (fridge.recognize(path, method), fridge.read_expiry(path), str(path))
        rec, exp, saved = st.session_state["result"]
        st.image(shot, width=320)
        st.caption(f"인식: {rec['name'] or '모름'} (점수 {rec['score']}, {rec['sec']}초) · "
                   f"유통기한: {exp['date'] or '못 읽음'} ({exp['sec']}초) · 근거: {exp['evidence'] or '-'}")
        name = st.selectbox("품목", fridge.NAMES, index=fridge.NAMES.index(rec["name"]) if rec["name"] else 0)
        guess = exp["date"] or default_expiry(name)
        exp_date = st.date_input("유통기한", value=guess, key=f"d_{st.session_state['shot_id']}")
        if st.button("등록", type="primary", disabled=exp_date is None):
            add(name, exp_date, "photo", saved)
            log("photo", time.perf_counter() - st.session_state["t0"], rec["name"], exp["date"], name, exp_date)
            st.success(f"{name} ({exp_date}) 등록")
            st.session_state.update(shot_id=None, result=None)

with tab_manual:
    if "m_t0" not in st.session_state:
        st.session_state["m_t0"] = time.perf_counter()
    m_name = st.selectbox("품목 ", fridge.NAMES)
    m_date = st.date_input("유통기한 ", value=None)
    if st.button("등록 ", disabled=m_date is None):
        add(m_name, m_date, "manual")
        log("manual", time.perf_counter() - st.session_state["m_t0"], "", "", m_name, m_date)
        st.success(f"{m_name} ({m_date}) 등록")
        st.session_state["m_t0"] = time.perf_counter()

with tab_list:
    with db() as con:
        rows = con.execute("select id, name, expiry, source from items where eaten=0 order by expiry").fetchall()
    if not rows:
        st.info("등록된 품목이 없습니다.")
    for iid, name, expiry, source in rows:
        left = (date.fromisoformat(expiry) - date.today()).days if expiry else None
        mark = "🔴 지남" if left is not None and left < 0 else "🟠 임박" if left is not None and left <= 1 else "🟢"
        c1, c2 = st.columns([4, 1])
        c1.write(f"{mark} **{name}** · {expiry or '날짜 없음'} · D{-left:+d}" if left is not None else f"{name} · 날짜 없음")
        if c2.button("먹음", key=f"eat{iid}"):
            with db() as con:
                con.execute("update items set eaten=1 where id=?", (iid,))
            st.rerun()
