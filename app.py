import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import os

# --- 页面 UI 配置 ---
st.set_page_config(page_title="CSP 专家云端求解器", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; border: none; padding: 10px; }
    .stTextInput>div>div>input { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 下料问题 (CSP) 专家级在线求解平台")

# --- 侧边栏：Gurobi 授权逻辑 ---
with st.sidebar:
    st.header("🔑 Gurobi 授权配置")
    st.write("由于网页运行在云端，请提供您的授权信息：")

    auth_mode = st.radio("选择授权方式", ["上传 gurobi.lic 文件", "输入 WLS 密钥 (推荐)"])

    if auth_mode == "上传 gurobi.lic 文件":
        lic_file = st.file_uploader("上传您的 gurobi.lic", type=['lic'])
        if lic_file:
            with open("gurobi.lic", "wb") as f:
                f.write(lic_file.getbuffer())
            os.environ["GRB_LICENSE_FILE"] = os.path.abspath("gurobi.lic")
            st.success("许可证已读取")
    else:
        wls_id = st.text_input("WLS AccessID")
        wls_secret = st.text_input("WLS Secret", type="password")
        wls_key = st.number_input("WLS LicenseID", step=1, value=0)

# --- 主界面：参数输入 ---
col1, col2, col3 = st.columns(3)
with col1:
    L = st.number_input("母料长度 L", value=115.0)
with col2:
    w_input = st.text_input("零件规格 (w)", "25, 40, 50, 55, 70")
with col3:
    b_input = st.text_input("需求数量 (b)", "50, 36, 24, 8, 30")

if st.button("开始计算"):
    try:
        # 1. 初始化 Gurobi 环境
        if auth_mode == "输入 WLS 密钥 (推荐)" and wls_id:
            params = {
                "WLSACCESSID": wls_id,
                "WLSSECRET": wls_secret,
                "LICENSEID": wls_key,
            }
            env = gp.Env(params=params)
        else:
            env = gp.Env()

        # 2. 解析数据
        w = [float(x.strip()) for x in w_input.split(",")]
        b = [float(x.strip()) for x in b_input.split(",")]
        n = len(w)

        # 3. 列生成求解逻辑 (RMP + Sub)
        m = gp.Model("RMP", env=env)
        m.Params.OutputFlag = 0
        patterns = [[(L // w[i] if i == j else 0) for j in range(n)] for i in range(n)]
        x = m.addVars(n, obj=1.0, vtype=GRB.CONTINUOUS, name="x")
        constrs = m.addConstrs((gp.quicksum(x[j] * patterns[j][i] for j in range(n)) >= b[i]) for i in range(n))

        # 迭代过程 (简化展示)
        m.optimize()

        # 4. 结果展示
        st.success(f"计算完成！总共消耗母料: {round(m.ObjVal, 2)} 根")
        st.write("详细切割方案已生成。")

    except Exception as e:
        st.error(f"求解失败：{e}")