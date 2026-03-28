import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd

# --- 页面 UI 配置 ---
st.set_page_config(page_title="CSP 专家云端求解器", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; border: none; padding: 10px; width: 100%; }
    .stTextInput>div>div>input { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 下料问题 (CSP) 列生成专家求解平台")

# --- 侧边栏：Gurobi 授权逻辑 ---
with st.sidebar:
    st.header("🔑 Gurobi 授权配置")
    auth_mode = st.radio("选择授权方式", ["上传 gurobi.lic 文件", "输入 WLS 密钥 (推荐)"])
    params = {}
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
        if wls_id and wls_secret and wls_key:
            params = {"WLSACCESSID": wls_id, "WLSSECRET": wls_secret, "LICENSEID": int(wls_key)}

# --- 主界面：参数输入 ---
col1, col2, col3 = st.columns(3)
with col1:
    L = st.number_input("母料长度 L", value=500.0)
with col2:
    w_input = st.text_input("零件规格 (w_i)", "45, 60, 75, 90, 110, 125, 140, 160, 180, 200, 225, 250, 280, 310, 350")
with col3:
    b_input = st.text_input("需求数量 (b_i)", "80, 120, 45, 60, 30, 90, 55, 40, 70, 25, 50, 35, 20, 15, 10")

if st.button("开始云端求解"):
    st.write("### 🔄 列生成迭代日志")
    log_area = st.container() 

    try:
        # 1. 解析数据
        w = [float(x.strip()) for x in w_input.split(",")]
        b = [float(x.strip()) for x in b_input.split(",")]
        n = len(w)

        # 2. 初始化环境
        env = gp.Env(params=params) if params else gp.Env()
        
        # --- 3. 初始化 RMP (LP 松弛) ---
        rmp = gp.Model("RMP", env=env)
        rmp.Params.OutputFlag = 0
        patterns = [[(L // w[i] if i == j else 0) for j in range(n)] for i in range(n)]
        x_vars = rmp.addVars(n, obj=1.0, vtype=GRB.CONTINUOUS, name="x")
        constrs = [rmp.addConstr(gp.quicksum(x_vars[j] * patterns[j][i] for j in range(n)) >= b[i], name=f"c_{i}") for i in range(n)]

        # --- 4. 列生成循环 ---
        while True:
            rmp.optimize()
            duals = [c.Pi for c in constrs]
            sub = gp.Model("Sub", env=env); sub.Params.OutputFlag = 0
            y = sub.addVars(n, vtype=GRB.INTEGER, name="y")
            sub.setObjective(gp.quicksum(duals[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
            sub.addConstr(gp.quicksum(w[i] * y[i] for i in range(n)) <= L)
            sub.optimize()
            reduced_cost = 1 - sub.ObjVal
            if reduced_cost >= -1e-6: break
            new_pattern = [int(y[i].X + 0.5) for i in range(n)]
            log_area.write(f"发现新列: `{new_pattern}` | 当前LP值: `{round(rmp.ObjVal, 2)}`")
            patterns.append(new_pattern)
            new_col = gp.Column(new_pattern, constrs)
            rmp.addVar(obj=1.0, column=new_col, vtype=GRB.CONTINUOUS, name=f"x_{len(patterns)-1}")

        # --- 5. MIP 最终整数求解与利用率计算 ---
        st.write("---")
        st.write("### 🏆 最终整数切割方案 (MIP)")
        for v in rmp.getVars(): v.vtype = GRB.INTEGER
        rmp.optimize()
        
        if rmp.status == GRB.OPTIMAL:
            total_used = int(rmp.ObjVal + 0.5)
            
            # 计算利用率
            total_required_length = sum(wi * bi for wi, bi in zip(w, b))
            total_provided_length = total_used * L
            efficiency = (total_required_length / total_provided_length) * 100

            # 顶部指标显示
            m_col1, m_col2 = st.columns(2)
            m_col1.metric("母料总需求量", f"{total_used} 根")
            m_col2.metric("原材料总利用率", f"{round(efficiency, 2)}%")

            results = []
            for j, v in enumerate(rmp.getVars()):
                if v.X > 0.5:
                    used_qty = int(v.X + 0.5)
                    pattern = patterns[j]
                    p_len = sum(pattern[k] * w[k] for k in range(n))
                    pattern_desc = " + ".join([f"{int(pattern[k])}个[{w[k]}mm]" for k in range(n) if pattern[k] > 0])
                    results.append({
                        "方案编号": f"模式 {j+1}",
                        "切割详情 (单根)": pattern_desc,
                        "使用根数": used_qty,
                        "单根剩余空间": f"{round(L - p_len, 2)} mm"
                    })
            st.dataframe(pd.DataFrame(results), use_container_width=True)
            st.success(f"💡 优化建议：通过列生成算法，您节省了大量原材料。当前方案的材料净长度为 {round(total_required_length, 2)}mm，总供应长度为 {total_provided_length}mm。")
        else:
            st.error("无法获得整数可行解。")

    except Exception as e:
        st.error(f"求解过程中出现错误: {e}")
