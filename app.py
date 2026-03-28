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
            params = {"WLSACCESSID": wls_id, "WLSSECRET": wls_secret, "LICENSEID": wls_key}

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
        
        # 初始模式：单位阵
        patterns = [[(L // w[i] if i == j else 0) for j in range(n)] for i in range(n)]
        x_vars = rmp.addVars(n, obj=1.0, vtype=GRB.CONTINUOUS, name="x")
        
        # 核心修复点：将约束存储在字典或列表中
        constrs = []
        for i in range(n):
            c = rmp.addConstr(gp.quicksum(x_vars[j] * patterns[j][i] for j in range(n)) >= b[i], name=f"c_{i}")
            constrs.append(c)

        # --- 4. 列生成循环 ---
        iter_count = 0
        while True:
            iter_count += 1
            rmp.optimize()
            
            # 正确获取对偶值 (Dual Prices)
            duals = [c.Pi for c in constrs]
            
            # 子问题：背包问题
            sub = gp.Model("Sub", env=env)
            sub.Params.OutputFlag = 0
            y = sub.addVars(n, vtype=GRB.INTEGER, name="y")
            sub.setObjective(gp.quicksum(duals[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
            sub.addConstr(gp.quicksum(w[i] * y[i] for i in range(n)) <= L)
            sub.optimize()
            
            reduced_cost = 1 - sub.ObjVal
            
            # 打印过程
            log_area.markdown(f"**第 {iter_count} 次迭代**: LP 目标值 = `{round(rmp.ObjVal, 2)}`, 检验数 = `{round(reduced_cost, 4)}`")

            # 停止条件
            if reduced_cost >= -1e-6:
                log_area.success("✨ 已达到 LP 最优，停止迭代。")
                break
            
            # 添加新切割方案 (New Column)
            new_pattern = [int(y[i].X + 0.5) for i in range(n)]
            log_area.markdown(f"&nbsp;&nbsp;&nbsp;&nbsp;➡️ 发现新列: `{new_pattern}`")
            patterns.append(new_pattern)
            
            # 将新列加入 RMP
            new_col = gp.Column(new_pattern, constrs)
            rmp.addVar(obj=1.0, column=new_col, vtype=GRB.CONTINUOUS, name=f"x_{len(patterns)-1}")

        # --- 5. MIP 最终整数求解 ---
        st.write("---")
        st.write("### 🏆 最终整数切割方案 (MIP)")
        
        # 将所有变量转为整数
        for v in rmp.getVars():
            v.vtype = GRB.INTEGER
        
        rmp.optimize()
        
        if rmp.status == GRB.OPTIMAL:
            st.metric("母料总需求量", f"{int(rmp.ObjVal + 0.5)} 根")
            
            results = []
            for j, v in enumerate(rmp.getVars()):
                if v.X > 0.5:
                    pattern_desc = ", ".join([f"{int(patterns[j][k])}个[{w[k]}mm]" for k in range(n) if patterns[j][k] > 0])
                    results.append({
                        "方案编号": f"模式 {j+1}",
                        "切割详情": pattern_desc,
                        "使用根数": int(v.X + 0.5)
                    })
            st.table(pd.DataFrame(results))
        else:
            st.error("无法获得整数可行解。")

    except Exception as e:
        st.error(f"求解过程中出现错误: {e}")
