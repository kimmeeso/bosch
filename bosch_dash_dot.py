import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 1. 페이지 설정
st.set_page_config(layout="wide", page_title="LMS 장애 진단 시연")

# 2. 데이터 로드 함수 (캐싱 적용)
@st.cache_data
def get_data(file_type):
    if file_type == "정상 운영 (Normal)":
        # 실제 경로의 파일 읽기: df = pd.read_csv('data_normal.csv')
        file_path = './data/엑셀data_20260203_161051.xlsx'
        df = pd.read_excel(file_path)
    else:
        # 실제 경로의 파일 읽기: df = pd.read_csv('data_abnormal.csv')
        file_path = './data/엑셀data_20260203_161051_adnormal.xlsx'
        df = pd.read_excel(file_path)

    return df

# 3. 최적화된 그래프 함수 (검정 선 + 이상치 빨간 점)
def draw_chart(df, keyword, title):
    target_cols = [c for c in df.columns if keyword.lower() in c.lower() and c != 'Time_ms']
    if not target_cols: 
        return st.write(f"{keyword} 변수가 없습니다.")

    fig = go.Figure()
    
    for col in target_cols:
        # (A) 전체 데이터를 스캐터(점)로 표시 - Scattergl 사용으로 성능 최적화
        fig.add_trace(go.Scattergl(
            x=df['Time_ms'], 
            y=df[col], 
            name=f"{col} (Raw)",
            mode='markers',
            marker=dict(
                color='black', 
                size=3,      # 점 크기를 작게 하여 밀집도를 표현
                opacity=0.6  # 투명도를 주면 데이터 겹침(밀도) 확인 용이
            ),
            hovertemplate=f'<b>{col}</b><br>값: %{{y}}<extra></extra>'
        ))

        # (B) PosError 한정 빨간 점 강조 (절대값 15,000 이상)
        if 'poserror' in col.lower():
            anomaly = df[df[col].abs() >= 15000]
            if not anomaly.empty:
                fig.add_trace(go.Scattergl(
                    x=anomaly['Time_ms'], 
                    y=anomaly[col],
                    name="⚠️ Critical Fault", 
                    mode='markers',
                    marker=dict(
                        color='red', 
                        size=7,       # 이상치는 더 크고 진하게
                        symbol='circle',
                        line=dict(width=1, color='DarkSlateGrey') # 테두리 추가로 강조
                    )
                ))

    # 레이아웃 설정
    fig.update_layout(
        title=title, 
        template="plotly_white", 
        height=350, 
        margin=dict(l=10, r=10, t=40, b=10),
        hovermode="x unified",
        # WebGL 성능 극대화를 위한 설정
        uirevision='constant' 
    )
    
    st.plotly_chart(fig, use_container_width=True)

# --- 사이드바: 데이터 선택 ---
with st.sidebar:
    st.header("🎮 시연 시나리오 선택")
    data_choice = st.radio(
        "현재 구동 모드 선택:",
        ("정상 운영 (Normal)", "비정상/장애 발생 (Abnormal)")
    )
    
    st.divider()
    if data_choice == "정상 운영 (Normal)":
        st.success("✅ 시스템 상태: 정상 (Healthy)")
        st.info("운영자 메시지: 현재 공정은 오차율 0%로 완벽하게 제어되고 있습니다.")
    else:
        st.error("🚨 시스템 상태: 장애 감지 (Critical)")
        st.warning("경고: Following Error Limit 초과. 시스템 보호를 위해 셧다운 권장.")

# --- 메인 화면 ---
st.title(f"🔍 LMS 실시간 분석 - {data_choice}")
df = get_data(data_choice)

# 2x2 배치 및 하단 강조 배치
c1, c2 = st.columns(2)
with c1:
    draw_chart(df, 'CarVel', "1. 속도 프로파일")
    draw_chart(df, 'PosError', "2. 위치 오차 (절대값 15,000 이상 강조)")
with c2:
    draw_chart(df, 'Pos_1', "3. 위치 트래킹 1")
    draw_chart(df, 'Pos_2', "4. 위치 트래킹 2")

st.divider()
draw_chart(df, 'CoilCurrent', "5. 코일 전류 분석 (최대 출력 포화 여부 확인)")


# streamlit run .\bosch_dash_dot.py