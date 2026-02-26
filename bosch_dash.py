import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import glob
import base64
import time
import altair as alt

# 고속(벡터화) 이슈 추출 함수 (파일 내장)



# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="Bosch Abnormal Diagnostic System", initial_sidebar_state="expanded")

# --- 세션 상태 초기화 ---
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "detected_issues" not in st.session_state:
    st.session_state.detected_issues = []
if "chat_open" not in st.session_state:
    st.session_state.chat_open = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "selected_issue_row" not in st.session_state:
    st.session_state.selected_issue_row = None
if "selected_issue_key" not in st.session_state:
    # (Time(ms), Variable)로 선택 이슈를 고정해 테이블 리렌더에도 흔들리지 않게 함
    st.session_state.selected_issue_key = None
if "chat_messages_by_issue" not in st.session_state:
    # issue_key(tuple) -> list[{"role": "...", "content": "..."}]
    st.session_state.chat_messages_by_issue = {}
if "wrapped" not in st.session_state:
    st.session_state.wrapped = False
if "issue_seen_keys" not in st.session_state:
    st.session_state.issue_seen_keys = set()
if "issue_gap_ms" not in st.session_state:
    st.session_state.issue_gap_ms = None
if "issue_run_state" not in st.session_state:
    # var -> {"idx": int, "count": int, "last_time": int}
    st.session_state.issue_run_state = {}
if "just_reset" not in st.session_state:
    st.session_state.just_reset = False
if "unread_issue_count" not in st.session_state:
    # '이슈 히스토리' 탭에 들어가기 전까지의 신규 이슈(연속 구간의 첫 사례) 개수
    st.session_state.unread_issue_count = 0
if "last_issue_summary" not in st.session_state:
    st.session_state.last_issue_summary = None
if "last_menu" not in st.session_state:
    st.session_state.last_menu = None
if "current_menu" not in st.session_state:
    st.session_state.current_menu = None
if "notif" not in st.session_state:
    # { "msg": str, "level": "warning"|"error", "expires_at": float }
    st.session_state.notif = None

# --- Live 재생 설정값(메뉴 이동해도 유지) ---
if "window_size" not in st.session_state:
    st.session_state.window_size = 35
if "step_size" not in st.session_state:
    st.session_state.step_size = 5
if "live_render_every" not in st.session_state:
    # 모든 tick마다 차트를 다시 그리면 버벅임이 생깁니다. N번 tick마다 1번만 렌더링.
    st.session_state.live_render_every = 1
if "live_tick_counter" not in st.session_state:
    st.session_state.live_tick_counter = 0
if "render_interval_sec" not in st.session_state:
    st.session_state.render_interval_sec = 0.7

# --- 세션 상태 추가 (상단 초기화 부분에 삽입) ---
if "selected_cols_dict" not in st.session_state:
    # 각 차트별로 선택된 컬럼을 개별 저장
    st.session_state.selected_cols_dict = {
        'CarVel_': [], 'Pos_1': [], 'Pos_2': [], 'CoilCurrent': [], 'PosError': []
    }

# --- [수정] st_autorefresh 임포트 부분 완전 삭제 ---
if "history_dirty" not in st.session_state:
    st.session_state.history_dirty = False  # [신규 추가] 새 에러 발생 여부 플래그

# 2. 데이터 로드 함수
@st.cache_data(show_spinner=False)
def get_abnormal_data():
    try:
        files = glob.glob(os.path.join("./data", "*adnormal*.csv"))
        if not files:
            files = glob.glob(os.path.join("./20.Data", "*adnormal*.csv"))
            # files = glob.glob(os.path.join("./data", "*adnormal*_level2.csv"))
        df = pd.read_csv(files[0])
        print(files[0])
        return df, os.path.basename(files[0])
    except:
        t = np.arange(0, 5000, 10)
        df = pd.DataFrame({'Time_ms': t})
        for i in range(1, 3):
            df[f'CoilCurrent{i:02d}'] = np.random.randn(len(t)) * 5
            df[f'PosError{i:02d}'] = np.random.randn(len(t)) * 100
            df[f'CarVel_{i:02d}'] = np.random.randn(len(t)) * 10
            df[f'Pos_{i:02d}'] = np.cumsum(np.abs(np.random.randn(len(t)) * 5))
        return df, "Simulation_Mode.xlsx"

# 3. 이슈 추출 함수(벡터화 고속 버전)
def _build_issue_events(df, cols, warn_th, fault_th, issue_type):
    if df is None or df.empty or (not cols) or ("Time_ms" not in df.columns):
        return pd.DataFrame(columns=["Time (ms)", "Variable", "Status", "Value", "Type"])

    sub = df[cols]
    abs_sub = sub.abs()

    mask = abs_sub.ge(warn_th)
    if not mask.to_numpy().any():
        return pd.DataFrame(columns=["Time (ms)", "Variable", "Status", "Value", "Type"])

    # pandas 버전별 stack 시그니처/동작 차이 대응
    try:
        stacked = sub.where(mask).stack(future_stack=True).dropna()
    except TypeError:
        stacked = sub.where(mask).stack(dropna=True)

    if stacked.empty:
        return pd.DataFrame(columns=["Time (ms)", "Variable", "Status", "Value", "Type"])

    events = stacked.reset_index()
    events.columns = ["_row_idx", "Variable", "_raw_value"]

    # Time_ms join (df의 index를 사용)
    time_map = df.loc[events["_row_idx"], "Time_ms"].to_numpy()
    events["Time (ms)"] = pd.to_numeric(time_map, errors="coerce")
    events = events.dropna(subset=["Time (ms)"])
    if events.empty:
        return pd.DataFrame(columns=["Time (ms)", "Variable", "Status", "Value", "Type"])
    events["Time (ms)"] = events["Time (ms)"].astype(int)

    abs_val = pd.to_numeric(events["_raw_value"], errors="coerce").abs()
    events["Status"] = np.where(
        abs_val.ge(fault_th),
        "🚨 Level 3: Fault",
        "⚠️ Level 2: Warning",
    )

    # Value는 기존과 동일하게 소수 2자리 문자열(부호 유지)
    events["Value"] = pd.to_numeric(events["_raw_value"], errors="coerce").map(lambda v: f"{v:.2f}")
    events["Type"] = issue_type

    return events[["Time (ms)", "Variable", "Status", "Value", "Type"]]


def extract_issues(df):
    """기존 extract_issues와 동일 포맷을 반환하되, 벡터화로 빠르게 추출."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["Time (ms)", "Variable", "Status", "Value", "Type"])

    cols = list(df.columns)
    coil_cols = [c for c in cols if "CoilCurrent" in c]
    err_cols = [c for c in cols if "PosError" in c]

    df_curr = _build_issue_events(df, coil_cols, warn_th=22, fault_th=25, issue_type="Current Limit")
    df_err = _build_issue_events(df, err_cols, warn_th=5000, fault_th=10000, issue_type="Pos Error")

    out = pd.concat([df_curr, df_err], ignore_index=True)
    if out.empty:
        return out

    return out.sort_values(by=["Time (ms)", "Variable"], kind="mergesort").reset_index(drop=True)

# 4. 차트 생성 함수
def create_chart_object(df_plot, keyword, title):
    # 1. 대상 컬럼 필터링 및 데이터 녹이기 (Melt)
    target_cols = [c for c in df_plot.columns if keyword.lower() in c.lower() and c != 'Time_ms']
    
    if not target_cols:
        return alt.Chart(pd.DataFrame()).mark_text(text="No Data").properties(title=title, height=300)

    df_long = df_plot.melt('Time_ms', value_vars=target_cols, var_name='Variable', value_name='Value')

    # 2. 임계값(Threshold) 및 Y축 고정 범위(Domain) 설정
    limit = None
    y_domain = None
    
    # 키워드별 Y축 범위와 한계선 세팅
    if 'coilcurrent' in keyword.lower(): 
        limit = 22
        y_domain = [-35, 35]
    elif 'poserror' in keyword.lower(): 
        limit = 5000
        y_domain = [-21000, 21000]
    elif 'vel' in keyword.lower(): 
        y_domain = [-5500, 5500]
    elif 'pos' in keyword.lower(): 
        y_domain = [-100, 4100]

    # Y축 스케일 객체 생성 (범위가 지정되었으면 고정, 아니면 자동)
    y_scale = alt.Scale(domain=y_domain, clamp=True) if y_domain else alt.Scale(zero=False)

    # ---------------------------------------------------------
    # [Layer 1] 메인 라인 차트
    # ---------------------------------------------------------
    base = alt.Chart(df_long).encode(
        x=alt.X('Time_ms', axis=alt.Axis(labels=False, title=None, tickCount=5)),
        y=alt.Y('Value', title=None, scale=y_scale), 
        
        # 🎨 [핵심 수정] scale=alt.Scale(scheme='category10') 추가!
        # 이렇게 하면 무조건 첫 번째 컬럼은 파랑(#1f77b4), 두 번째는 주황(#ff7f0e)으로 고정됩니다.
        color=alt.Color(
            'Variable', 
            scale=alt.Scale(scheme='category10'), 
            legend=alt.Legend(orient='bottom', title=None)
        ), 
        
        tooltip=['Time_ms', 'Variable', 'Value']
    )
    line_layer = base.mark_line(interpolate='linear', strokeWidth=2)
    
    # ---------------------------------------------------------
    # [Layer 2 & 3] 가이드라인과 🚨 빨간 점 (임계값이 있을 때만)
    # ---------------------------------------------------------
    if limit:
        # 상/하한선 (점선)
        rule_up = alt.Chart(pd.DataFrame({'y': [limit]})).mark_rule(
            strokeDash=[4, 4], color='orange', size=1
        ).encode(y='y')
        rule_down = alt.Chart(pd.DataFrame({'y': [-limit]})).mark_rule(
            strokeDash=[4, 4], color='orange', size=1
        ).encode(y='y')
        
        layers.extend([rule_up, rule_down])

        # 에러 포인트 (빨간 점)
        points = base.transform_filter(
            (alt.datum.Value >= limit) | (alt.datum.Value <= -limit)
        ).mark_circle(size=60, color='red', opacity=1)
        
        layers.append(points)

    # ---------------------------------------------------------
    # 렌더링 조합 및 레이아웃 설정
    # ---------------------------------------------------------
    combined_chart = alt.layer(*layers).properties(
        title=title,
        height=320 # 차트 높이
    ).configure_axis(
        grid=True, gridOpacity=0.3
    ).configure_title(
        fontSize=15, anchor='start', color='#333'
    )

    return combined_chart
    
    # Y축 범위 설정 (기존 로직 유지)
    y_range = None
    if 'coilcurrent' in keyword.lower(): y_range = [-35, 35]
    elif 'poserror' in keyword.lower(): y_range = [-21000, 21000]
    elif 'vel' in keyword.lower(): y_range = [-5500, 5500]
    elif 'pos' in keyword.lower(): y_range = [-100, 4100]

# (기존 코드) Y축 범위 설정 부분 아래의 update_layout을 이렇게 교체하세요.
    
    fig.update_layout(
        title=dict(text=title), template="plotly_white", height=320, 
        margin=dict(l=10, r=10, t=45, b=10),
        
        # 🔒 1. Y축 고정: autorange 대신 명시적 range 사용, 연산 차단(fixedrange)
        yaxis=dict(
            range=y_range if y_range else [df_plot[display_cols].min().min(), df_plot[display_cols].max().max()],
            fixedrange=True 
        ),
        
        # 🔒 2. X축 고정: 현재 데이터 윈도우의 시작과 끝을 절대값으로 고정
        xaxis=dict(
            title=dict(text="Time (ms)"), 
            showticklabels=False, showgrid=False, zeroline=False, showline=False,
            range=[df_plot['Time_ms'].min(), df_plot['Time_ms'].max()],
            fixedrange=True
        ),
        
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        uirevision=str(keyword),
        
        # 🔒 3. 애니메이션 끄기 (스트림릿에서는 애니메이션이 오히려 껌벅임을 유발함)
        transition_duration=0
    )
    return fig



# 5. 로컬 이미지를 웹에서 읽을 수 있도록 변환하는 함수
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

df_full, file_name = get_abnormal_data()

def _infer_time_step_ms(df: pd.DataFrame) -> int:
    if df is None or 'Time_ms' not in df.columns or len(df) < 3:
        return 10
    try:
        diffs = pd.to_numeric(df['Time_ms'], errors='coerce').diff().dropna()
        diffs = diffs[diffs > 0]
        if diffs.empty:
            return 10
        return int(diffs.median())
    except Exception:
        return 10

TIME_STEP_MS = _infer_time_step_ms(df_full)
if st.session_state.issue_gap_ms is None:
    st.session_state.issue_gap_ms = TIME_STEP_MS

def reset_issue_history():
    st.session_state.detected_issues = []
    st.session_state.issue_seen_keys = set()
    st.session_state.issue_run_state = {}
    st.session_state.selected_issue_row = None
    st.session_state.last_selected_issue = None
    st.session_state.chat_open = False
    st.session_state.messages = []
    st.session_state.selected_issue_key = None
    st.session_state.chat_messages_by_issue = {}
    st.session_state.unread_issue_count = 0
    st.session_state.last_issue_summary = None

def _append_unique_issues(sub_issues: pd.DataFrame) -> int:
    """연속 이슈는 첫 건만 남기고, 반복 횟수(Count)를 누적."""
    if sub_issues is None or sub_issues.empty:
        return 0

    added = 0

    # 시간순 처리 (extract_issues가 slice 순서 유지하지만, 안전하게 정렬)
    try:
        sub_issues = sub_issues.sort_values(by="Time (ms)")
    except Exception:
        pass

    gap_ms = int(TIME_STEP_MS or 10) if st.session_state.issue_gap_ms is None else int(st.session_state.issue_gap_ms)

    for _, issue in sub_issues.iterrows():
        t = issue.get('Time (ms)')
        var = str(issue.get('Variable', ''))
        try:
            t_int = int(float(t))
        except Exception:
            continue

        # (1) 완전 중복 방지: 동일 Time, 동일 Variable (윈도우 겹침으로 인한 중복 방지)
        key = (t_int, var)
        if key in st.session_state.issue_seen_keys:
            continue
        st.session_state.issue_seen_keys.add(key)

        # (2) 플래그 기반 연속 이슈 압축
        state = st.session_state.issue_run_state.get(var)
        if state is not None and (t_int - int(state.get("last_time", -10**18))) <= gap_ms:
            # 연속 구간: 첫 건은 유지, count만 증가
            state["count"] = int(state.get("count", 1)) + 1
            state["last_time"] = t_int
            idx = int(state.get("idx", -1))
            if 0 <= idx < len(st.session_state.detected_issues):
                rec = st.session_state.detected_issues[idx]
                rec["Repeat Count"] = state["count"]
                rec["Last Time (ms)"] = t_int
        else:
            # 신규 구간 시작: 첫 건을 기록하고 상태 생성
            rec = issue.to_dict()
            rec["Repeat Count"] = 1
            rec["Last Time (ms)"] = t_int
            st.session_state.detected_issues.append(rec)
            st.session_state.issue_run_state[var] = {"idx": len(st.session_state.detected_issues) - 1, "count": 1, "last_time": t_int}
            added += 1
            
            # --- [신규 추가] 새 에러가 발생했음을 표시 ---
            st.session_state.history_dirty = True 

            # 알림(읽지 않은 신규 이슈) 집계: 히스토리 탭이 아닐 때만 누적
            st.session_state.last_issue_summary = {
                "Time (ms)": t_int,
                "Variable": var,
                "Status": rec.get("Status", ""),
                "Type": rec.get("Type", ""),
            }
            if st.session_state.current_menu != "이슈 히스토리":
                st.session_state.unread_issue_count = int(st.session_state.unread_issue_count) + 1
    return added

def _is_action_request(text: str) -> bool:
    t = (text or "").replace(" ", "")
    keywords = [
        "조치", "조치방안", "조치방법", "대응", "대응방안", "가이드", "알려줘", "알려주세요", "어떻게",
        "원인", "해결", "해결방법", "대처",
    ]
    return any(k in t for k in keywords)


def _push_notif(msg: str, level: str = "warning", seconds: float = 5.0) -> None:
    """오른쪽 상단 커스텀 알림을 seconds 동안 유지."""
    st.session_state.notif = {
        "msg": msg,
        "level": level,
        "expires_at": time.time() + float(seconds),
    }


def _render_notif() -> None:
    """알림이 살아있으면 오른쪽 상단에 고정 표시."""
    n = st.session_state.notif
    if not n:
        return
    if time.time() >= float(n.get("expires_at", 0)):
        st.session_state.notif = None
        return

    level = n.get("level", "warning")
    bg = "#fff3cd" if level == "warning" else "#f8d7da"
    border = "#ffeeba" if level == "warning" else "#f5c6cb"
    color = "#856404" if level == "warning" else "#721c24"

    st.markdown(
        f"""
        <style>
          .notif-fixed {{
            position: fixed;
            top: 130px;
            right: 90px;
            z-index: 9999;
            max-width: 420px;
            padding: 10px 12px;
            border: 1px solid {border};
            background: {bg};
            color: {color};
            border-radius: 10px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.12);
            font-size: 14px;
          }}
        </style>
        <div class="notif-fixed">{n.get("msg","")}</div>
        """,
        unsafe_allow_html=True,
    )

def live_tick(window_size: int, step_size: int, notify: bool) -> pd.DataFrame:
    """무한루프 없이 '한 스텝'만 진행. 어떤 메뉴에서도 호출 가능."""
    # 데이터 끝까지 갔다가 다시 0으로 돌아오는 순간, 히스토리도 초기화
    if int(st.session_state.current_idx) == 0 and bool(st.session_state.wrapped):
        st.session_state.just_reset = True
        reset_issue_history()
        st.session_state.wrapped = False

    max_start = max(0, len(df_full) - window_size)
    i = int(st.session_state.current_idx)
    if i > max_start:
        i = 0

    df_sub = df_full.iloc[i : i + window_size]

    sub_issues = extract_issues(df_sub)
    added = _append_unique_issues(sub_issues)

    if notify and added > 0:
        # 새 이슈가 다수일 수 있으니 가장 최근 1건만 토스트
        last = st.session_state.detected_issues[-1]
        status = str(last.get("Status", ""))
        var = str(last.get("Variable", ""))
        if "Level 3" in status:
            _push_notif(f"🚨 Fault 감지: <b>{var}</b>", level="error", seconds=5.0)
        else:
            _push_notif(f"⚠️ Warning 감지: <b>{var}</b>", level="warning", seconds=5.0)

    next_i = i + step_size
    if max_start > 0 and next_i > max_start:
        st.session_state.current_idx = 0
        st.session_state.wrapped = True
    else:
        st.session_state.current_idx = next_i if max_start > 0 else 0

    st.session_state.live_tick_counter = int(st.session_state.live_tick_counter) + 1
    return df_sub


try:
    image_base64 = get_base64_image("./logo/logo_bosch.png") # 실제 파일명 입력
except:
    image_base64 = "" # 파일이 없을 경우 대비

# --- 사이드바 ---
with st.sidebar:
    # 사이드바 최상단에 로고 이미지 삽입 (반응형 크기)
    st.markdown("""
    <style>
    .sidebar-logo-wrapper {
        max-width: clamp(100px, 90%, 200px);
        margin: 0 auto 10px auto;
    }
    .sidebar-logo-wrapper img {
        width: 100%;
        height: auto;
        display: block;
        object-fit: contain;
    }
    </style>
    """, unsafe_allow_html=True)
    
    try:
        # 상사님이 주신 로고 이미지 경로 (반응형)
        logo_skax_base64 = get_base64_image("./logo/logo_skax.png")
        st.markdown(
            f'<div class="sidebar-logo-wrapper"><img src="data:image/png;base64,{logo_skax_base64}"></div>',
            unsafe_allow_html=True
        )
    except:
        try:
            st.image("./logo/logo_skax.png", width=120)
        except:
            st.title("🛡️ Bosch LMS")
            st.caption("Advanced Diagnostic System")

    st.divider() # 로고 아래 구분선 추가로 깔끔하게 정리

    st.title("🎮 제어 메뉴")
    menu = st.radio("이동", ["현황 정보 (Live)", "이슈 히스토리"])
    st.session_state.current_menu = menu

    # 이슈 히스토리에 들어가면 '읽음 처리'로 알림 해제
    if menu == "이슈 히스토리" and st.session_state.last_menu != "이슈 히스토리":
        st.session_state.unread_issue_count = 0
    st.divider()

    # Live 실행/설정은 메뉴와 무관하게 항상 표시 (이슈 히스토리에서도 Live를 '뒤에서' 돌릴 수 있음)
    c1, c2 = st.columns(2)
    if c1.button("🚀 시작", use_container_width=True, type="primary"):
        st.session_state.is_running = True
    if c2.button("⏹️ 중지", use_container_width=True):
        st.session_state.is_running = False
    if st.button("🔄 초기화", use_container_width=True):
        st.session_state.current_idx, st.session_state.is_running = 0, False
        reset_issue_history()
        st.session_state.wrapped = False
        st.session_state.just_reset = False
        st.rerun()
    
    st.divider()

    st.session_state.window_size = st.slider(
        "화면 데이터 수", 10, 100, int(st.session_state.window_size), 1
    )
    st.session_state.step_size = st.slider(
        "진행 보폭(step)", 1, 20, int(st.session_state.step_size), 1
    )
    st.session_state.render_interval_sec = st.slider(
        "그래프 갱신 주기(초)",
        0.3,
        2.0,
        float(st.session_state.render_interval_sec),
        0.1,
        help="너무 낮추면(빠르면) 렌더링 부하로 끊길 수 있습니다. 보통 0.5~1.0초가 무난합니다.",
    )

    st.divider()

    # --- 알림(읽지 않은 신규 이슈) 표시: 탭 이동과 무관하게 항상 보이게 ---
    if int(st.session_state.unread_issue_count) > 0:
        st.warning(f"🔔 새 이슈 {int(st.session_state.unread_issue_count)}건")
        if st.session_state.last_issue_summary:
            li = st.session_state.last_issue_summary
            st.caption(f"최근: `{li.get('Variable')}` @ {li.get('Time (ms)')}ms · {li.get('Status')}")

    # 2. 이미지 경로 설정 (로컬 logo 폴더 내 파일명으로 수정하세요)
# 3. [핵심] 탭 전환에도 흔들리지 않는 하단 고정 CSS 및 HTML
    if image_base64:
            st.markdown(
                f"""
                <style>
                /* 사이드바 여백 확보 */
                [data-testid="stSidebarUserContent"] {{
                    padding-top: 0vw; /* 기존 15vw에서 5vw로 감소 */
                    padding-bottom: 40vw; /* 하단 로고와 슬라이더 간 간격 증가 */
                }}
                /* 상단 로고 위치 및 크기 조정 */
                .sidebar-logo-top {{
                    max-width: clamp(120px, 15vw, 250px);
                    margin: 0 auto 20px auto;
                }}
                .sidebar-logo-top img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    object-fit: contain;
                }}
                /* 하단 로고 위치 및 크기 조정 */
                .sidebar-logo-bottom {{
                    position: absolute;
                    bottom: clamp(20px, 4vh, 50px);
                    left: 50%;
                    transform: translateX(-50%);
                    max-width: clamp(120px, 15vw, 250px);
                    z-index: 10;
                }}
                .sidebar-logo-bottom img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    object-fit: contain;
                }}
                </style>
                <div class="sidebar-footer-fixed">
                    <img src="data:image/png;base64,{image_base64}">
                    <p>LMS Diagnostic Reference</p>
                </div>
                """,
                unsafe_allow_html=True
            )

SUPPORTS_FRAGMENT = hasattr(st, "fragment")

MONITOR_INTERVAL_SEC = 0.2
RENDER_INTERVAL_SEC = float(st.session_state.render_interval_sec)  # Live 탭 그래프 렌더링 주기(초)

# 알림 렌더(오른쪽 상단): fragment 기반으로 주기 렌더링해서
# toast만 보이고 overlay가 안 보이는 문제를 방지
if SUPPORTS_FRAGMENT:
    @st.fragment(run_every=0.5)
    def _notif_fragment():
        _render_notif()

    _notif_fragment()
else:
    _render_notif()

# --- 백그라운드 모니터(렌더링 없이 상태만 갱신) ---
# Live는 뒤에서 계속 진행하되, "화면이 업데이트되는 모습"은 이슈 히스토리 탭에서만 보이도록 합니다.
if SUPPORTS_FRAGMENT:
    @st.fragment(run_every=(MONITOR_INTERVAL_SEC if st.session_state.is_running else None))
    def _issue_monitor_fragment():
        if not st.session_state.is_running:
            return

        live_tick(
            window_size=int(st.session_state.window_size),
            step_size=int(st.session_state.step_size),
            # 새 이슈(연속 구간의 첫 사례) 발생 시 짧은 토스트 알림
            # (live_tick 내부에서 added>0일 때 1건만 toast)
            notify=(not st.session_state.chat_open),
        )

        # wrap으로 자동 초기화가 일어났으면 플래그만 내려둠(표는 히스토리 탭에서 갱신 시 반영)
        if bool(st.session_state.just_reset):
            st.session_state.just_reset = False
            
        # --- [신규 추가] 에러 발생 시에만 이슈 히스토리 화면 갱신 ---
        if st.session_state.history_dirty:
            # 사용자가 이슈 히스토리 탭을 보고 있고, 채팅 중이 아닐 때만 리런
            if st.session_state.current_menu == "이슈 히스토리" and not st.session_state.chat_open:
                st.session_state.history_dirty = False # 리런 전 플래그 초기화
                st.rerun()

    _issue_monitor_fragment()

if menu == "현황 정보 (Live)":
    st.header("📊 실시간 분석 스트리밍")
    
    # --- 그래프별 필터 UI 추가 (루프 밖에서 정의하여 선택 유지) ---
    with st.expander("🎯 그래프 필터 설정 (보고 싶은 데이터만 선택)", expanded=False):
        f_col1, f_col2, f_col3 = st.columns(3)
        # CoilCurrent01~12, PosError01~12 등 전체 목록에서 선택
        st.session_state.selected_cols_dict['CoilCurrent'] = f_col1.multiselect(
            "Coil Current 선택", [c for c in df_full.columns if 'CoilCurrent' in c], key="ms_curr"
        )
        st.session_state.selected_cols_dict['PosError'] = f_col2.multiselect(
            "Position Error 선택", [c for c in df_full.columns if 'PosError' in c], key="ms_err"
        )

    if SUPPORTS_FRAGMENT:
        @st.fragment(run_every=(RENDER_INTERVAL_SEC if st.session_state.is_running else None))
        def _live_fragment():
            # --- Live 화면 상단 알림 배너 (이슈 히스토리 방문 시 자동 해제됨) ---
            if int(st.session_state.unread_issue_count) > 0:
                li = st.session_state.last_issue_summary or {}
                st.warning(
                    f"🔔 새 이슈 {int(st.session_state.unread_issue_count)}건 · "
                    f"최근: `{li.get('Variable','')}` @ {li.get('Time (ms)','')}ms · {li.get('Status','')}"
                )

            # Live 탭에서는 그래프가 계속 흐르도록 주기 렌더링합니다.
            # 데이터 진행(tick)은 백그라운드 monitor fragment가 담당합니다.
            i = int(st.session_state.current_idx)
            df_sub = df_full.iloc[i : i + int(st.session_state.window_size)]

    # (기존 _live_fragment 내부 수정)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                # 🟢 st.altair_chart 사용!
                st.altair_chart(create_chart_object(df_sub, 'CarVel_', "LMS Carrier 1&2 Velocity"), use_container_width=True)
            with col2:
                st.altair_chart(create_chart_object(df_sub, 'Pos_1', "LMS Position 1"), use_container_width=True)
            with col3:
                st.altair_chart(create_chart_object(df_sub, 'Pos_2', "LMS Position 2"), use_container_width=True)
    
            st.altair_chart(create_chart_object(df_sub, 'CoilCurrent', "LMS Coil Current"), use_container_width=True)
            st.altair_chart(create_chart_object(df_sub, 'PosError', "LMS Position Error"), use_container_width=True)                        
                    
            
            if st.session_state.is_running:
                st.info(f"Live 실행 중입니다. 그래프는 약 {RENDER_INTERVAL_SEC:.1f}초마다 갱신됩니다.")
            else:
                st.info(f"현재 {st.session_state.current_idx}ms 지점에서 대기 중입니다.")

        _live_fragment()

    else:
        # (구버전 Streamlit) fragment 미지원: ...
        if st.session_state.is_running:
            st.warning("현재 Streamlit 버전에서는 '이슈 발생 시에만 갱신'이 제한적입니다. Streamlit 업데이트를 권장합니다.")

        i = int(st.session_state.current_idx)
        df_sub = df_full.iloc[i : i + int(st.session_state.window_size)]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.plotly_chart(create_chart_object(df_sub, 'CarVel_', "LMS Carrier 1&2 Velocity"), use_container_width=True, key="wait_chart_vel")
        with col2:
            st.plotly_chart(create_chart_object(df_sub, 'Pos_1', "LMS Position 1"), use_container_width=True, key="wait_chart_pos1")
        with col3:
            st.plotly_chart(create_chart_object(df_sub, 'Pos_2', "LMS Position 2"), use_container_width=True, key="wait_chart_pos2")

        st.plotly_chart(create_chart_object(df_sub, 'CoilCurrent', "LMS Coil Current"), use_container_width=True, key="wait_chart_coil")
        st.plotly_chart(create_chart_object(df_sub, 'PosError', "LMS Position Error"), use_container_width=True, key="wait_chart_error")

        if st.session_state.is_running:
            st.info("Live 실행 중입니다. 화면은 '이슈 발생 시'에만 갱신됩니다.")
        else:
            st.info(f"현재 {st.session_state.current_idx}ms 지점에서 대기 중입니다.")

elif menu == "이슈 히스토리":
    # 헤더가 테이블보다 늦게 보이는 체감을 줄이기 위해,
    # 헤더/테이블을 고정 슬롯에 먼저 렌더링합니다.
    header_slot = st.empty()
    table_slot = st.empty()
    header_slot.header("📋 Detected Issue History (Real-time Updated)")

    if st.session_state.detected_issues:
        issue_df = pd.DataFrame(st.session_state.detected_issues)
        issue_df.insert(0, "No.", range(1, len(issue_df) + 1))
        event = table_slot.dataframe(
            issue_df,
            use_container_width=True,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun",
            key="issue_history_table",
        )
        if len(event.selection.rows) > 0:
            st.session_state.selected_issue_row = event.selection.rows[0]
            sel_row = issue_df.iloc[int(st.session_state.selected_issue_row)]
            st.session_state.selected_issue_key = (int(sel_row["Time (ms)"]), str(sel_row["Variable"]))
        else:
            st.session_state.selected_issue_row = None
            st.session_state.selected_issue_key = None
    else:
        table_slot.info("아직 감지된 이슈가 없습니다.")

    # 아래(세부 Plotly / 리포트 영역)는 자동 갱신에서 제외되어 '렌더링 느낌'을 최소화
    if st.session_state.selected_issue_row is not None and st.session_state.detected_issues:
        issue_df_detail = pd.DataFrame(st.session_state.detected_issues)
        issue_df_detail.insert(0, "No.", range(1, len(issue_df_detail) + 1))

        row = None
        # (우선) key 기반으로 선택 이슈를 찾음: 테이블 리렌더/행 이동에도 안정적
        if st.session_state.selected_issue_key is not None:
            t_key, v_key = st.session_state.selected_issue_key
            matched = issue_df_detail[
                (issue_df_detail["Time (ms)"] == t_key) & (issue_df_detail["Variable"].astype(str) == str(v_key))
            ]
            if not matched.empty:
                row = matched.iloc[0]

        # (fallback) 기존 row index 기반
        if row is None:
            sel = int(st.session_state.selected_issue_row)
            if 0 <= sel < len(issue_df_detail):
                row = issue_df_detail.iloc[sel]

        if row is not None:
            # --- [핵심 수정] 이슈 변경 감지 및 채팅 초기화 ---
            # 현재 선택한 이슈의 고유 키(여기선 Time_ms 사용)를 세션에 저장된 것과 비교
            current_issue_key = (int(row["Time (ms)"]), str(row["Variable"]))
            
            if "last_selected_issue" not in st.session_state:
                st.session_state.last_selected_issue = None

            # 이전에 선택했던 이슈와 지금 선택한 이슈가 다르면 채팅방 청소
            if st.session_state.last_selected_issue != current_issue_key:
                # 이슈별로 대화 기록을 분리 저장해서 테이블 리렌더와 상관없이 안정적으로 유지
                st.session_state.messages = st.session_state.chat_messages_by_issue.get(current_issue_key, [])
                st.session_state.last_selected_issue = current_issue_key # 현재 이슈 키 업데이트
                st.session_state.chat_open = False # 채팅창도 일단 닫음 (선택 사항)
            else:
                # 동일 이슈면 현재 세션 messages를 최신으로 동기화
                st.session_state.messages = st.session_state.chat_messages_by_issue.get(current_issue_key, st.session_state.messages)

            target_time = row['Time (ms)']
            idx_list = df_full.index[df_full['Time_ms'] == target_time].tolist()
            
            if idx_list:
                idx = idx_list[0]
                # 장애 시점 전후 10건 슬라이싱
                df_detail = df_full.iloc[max(0, idx - 10) : min(len(df_full), idx + 11)]

                st.divider()
                l_col, r_col = st.columns([0.6, 0.4]) if st.session_state.chat_open else st.columns([0.99, 0.01])

                with l_col:
                    st.subheader(f"🔍 세부 분석: {row['Variable']} (정밀 분석 모드)")
                    fig_d = go.Figure()
                    fig_d.add_trace(go.Scattergl(
                        x=df_detail['Time_ms'], 
                        y=df_detail[row['Variable']], 
                        mode='lines+markers', 
                        line=dict(color='red', width=2), 
                        marker=dict(size=8, color='red')
                    ))
                    
                    # Y축 범위 및 가이드라인 설정
                    y_range = [-35, 35] if "Current" in row['Variable'] else [-21000, 21000] if "Error" in row['Variable'] else None
                    limits = [(22, "Warn", "orange"), (25, "Fault", "red")] if "Current" in row['Variable'] else [(5000, "Warn", "orange"), (10000, "Fault", "red")] if "Error" in row['Variable'] else []
                    
                    for val, name, clr in limits:
                        fig_d.add_hline(y=val, line_dash="dot", line_color=clr, line_width=1, annotation_text=name)
                        fig_d.add_hline(y=-val, line_dash="dot", line_color=clr, line_width=1)

                    fig_d.update_layout(template="plotly_white", height=400, yaxis=dict(range=y_range))
                    st.plotly_chart(fig_d, use_container_width=True)
                    
                    # --- [디자인 개선 버전] Detailed Description ---
                    
                    st.markdown("### 📝 Detailed Analysis Report")

                    # 1. 이슈 유형 및 레벨 판별
                    is_current = "Current" in row['Variable']
                    is_level2 = "Level 2" in row['Status']
                    status_color = "warning" if is_level2 else "error"
                    icon = "⚠️" if is_level2 else "🚨"

                    # 2. [이미지 원문 100% 반영] 데이터 구성
                    if is_current:
                        title = "Bosch LMS Coil Current 에러 원인 및 대응 방안"
                        causes = [
                            "**제어 파라미터 부적합**: 과도한 P 게인 또는 급격한 명령으로 과전류 유도",
                            "**기구 마찰/간섭 증가**: LM 가이드 오염, 블록 손상, 정렬 불량 등으로 마찰/간섭이 커져 특정 구간 전류 상승",
                            "**전원/배선 이상**: uvw 케이블 접속 불량, 접지/쉴드 문제, 노이즈 유입으로 전류 리플/피크 증가",
                            "**부하 변화**: 페이로드 증가, 충돌/끼임 등 외란으로 토크 급상승해 평균/피크 전류 크게 증가"
                        ]
                        level2_res = "- 제어 파라미터 조정\n- 속도(가속/감속) 제한 하향 조정\n- 전류 제한 Limit 설정 강화\n- Proportional Gain (P-Gain)을 낮춰 과전류 유발 억제, 필요시 D-Gain 보강"
                        level3_res = "- 전원/기구/배선 점검\n- uvw 파워 케이블 라인 접속/단선/피복/접지 등 점검\n- LM 가이드, 블록, 캐리어 간섭/마찰/정렬 이상 점검 및 정비"
                    else:
                        title = "Bosch LMS Position 에러 원인 및 대응 방안"
                        causes = [
                            "**제어 파라미터 부적합**: 과도한 p 게인 또는 부족한 D/I 보상으로 추종 오차 증가",
                            "**과도한 속도/가속도 명령**: 모터 한계/기구 한계를 넘어 추종 불가",
                            "**기구물 문제**: LM 가이드 오염/마모, 평탄도 불량, 단차, 블록 유격/손상",
                            "**센서/스케일 문제**: 리니어 스케일 오염, 센서 신호 노이즈/단선",
                            "**부하 변화**: 캐리어 하중 증가, 마찰 급증(이물질, 윤활 부족)"
                        ]
                        level2_res = "- 제어 파라미터 조정\n- 속도(가속/감속) 제한 하향 조정\n- 전류 제한 Limit 설정 강화\n- Proportional Gain (P-Gain)을 낮춰 과전류 유발 억제, 필요시 D-Gain 보강"
                        level3_res = "- 기구물 및 센서 전수 점검\n- 캐리어, LM 가이드(수평/수직/평탄도/단차) 정밀 점검\n- LM 블록 상태(유격, 변형, 윤활) 점검/교체\n- 리니어 스케일, 센서 청소 및 고정 상태 확인"

                    # 3. 시각적 레이아웃 적용
                    with st.container(border=True):
                        # 제목을 더 크고 진하게 표시
                        st.markdown(f"## {icon} {title}")
                        
                        col_a, col_b = st.columns([1, 1])
                        with col_a:
                            st.markdown("#### 🔍 추정 원인 분석")
                            for c in causes:
                                # st.caption 대신 일반 markdown으로 굵게 표시하여 흐릿함을 방지
                                st.markdown(f"- {c}")
                                
                        with col_b:
                            st.markdown(f"#### 📍 현재 상태: **{row['Status']}**")
                            if is_level2:
                                st.warning("**Level 2 (Warning)**: 모니터링 지속 필요 (피크/평균 전류 확인)")
                            else:
                                st.error("**Level 3 (Fault)**: 시스템 보호를 위해 구동 정지 권고")

                    # 4. AI 버튼 (중앙 배치 및 강조)
                    st.write("") 
                    if not st.session_state.chat_open:
                        _, btn_col, _ = st.columns([0.1, 0.8, 0.1])
                        with btn_col:
                            if st.button("💬 보쉬 AI에게 상세 조치 방법 가이드 받기", use_container_width=True, type="primary"):
                                st.session_state.chat_open = True
                                st.rerun()

                    # 5. [개선된 디자인] AI Assistant 섹션
                    if st.session_state.chat_open:
                        with r_col:
                            st.info("🤖 **Bosch AI Assistant가 대응 매뉴얼을 분석 중입니다.**")
                            chat_container = st.container(border=True, height=550) 
                            
                            with chat_container:
                                header_col1, header_col2 = st.columns([0.8, 0.2])
                                header_col1.markdown("### 🤖 Bosch AI Assistant")
                                if header_col2.button("닫기", key="c_btn"):
                                    st.session_state.chat_open = False
                                    st.rerun()
                                
                                st.divider()
                                
                                # 첫 기본 메시지
                                with st.chat_message("assistant", avatar="🤖"):
                                    st.write(f"**{row['Variable']}**의 **{row['Status']}** 상태에 대한 대응 방안을 안내해 드립니다.")

                                # 대화 기록
                                for message in st.session_state.messages:
                                    with st.chat_message(message["role"], avatar="👤" if message["role"]=="user" else "🤖"):
                                        st.markdown(message["content"])

                            # 입력창
                            if prompt := st.chat_input("위 상황에 대한 조치 방법을 알려줘"): 
                                st.session_state.messages.append({"role": "user", "content": prompt})
                                
                                if _is_action_request(prompt):
                                    res = f"### 🛠️ {row['Status']} 대응 매뉴얼 원문\n"
                                    res += level2_res if is_level2 else level3_res
                                    res += "\n\n---\n*추가 점검이 필요하시면 현장 관리자에게 보고하십시오.*"
                                else:
                                    res = "이미지 매뉴얼에 따라 관련 파라미터 및 기구부를 점검하십시오. 상세 조치가 궁금하시면 '조치 방법'을 물어봐주세요."

                                st.session_state.messages.append({"role": "assistant", "content": res})
                                # 이슈별로 메시지 저장(테이블 리렌더/자동갱신과 무관)
                                st.session_state.chat_messages_by_issue[current_issue_key] = list(st.session_state.messages)
                                st.rerun()

# 메뉴 상태 기억(다음 rerun에서 탭 진입 감지용)
st.session_state.last_menu = st.session_state.current_menu








