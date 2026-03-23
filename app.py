import streamlit as st
import gurobipy as gp
from gurobipy import GRB
import os
import pandas as pd

# --- 页面 UI 配置 (保持美观) ---
st.set_page_config(page_title="CSP 专家云端求解器", layout="wide")
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { background-color: #4f46e5; color: white; border-radius: 8px; border: none; padding: 10px; width: 100%; }
    .stTextInput>div>div>input { border-radius: 8px; }
    .stAlert { border-radius: 8px; }
    </style>
    """, unsafe_allow_html=True)

st.title("🚀 下料问题 (CSP) 专家级在线求解平台")

# --- 侧边栏：Gurobi 授权逻辑 (保持不变) ---
with st.sidebar:
    st.header("🔑 Gurobi 授权配置")
    st.write("请提供您的授权信息以启用计算：")
    
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
            params = {
                "WLSACCESSID": wls_id,
                "WLSSECRET": wls_secret,
                "LICENSEID": wls_key,
            }

# --- 主界面：参数输入 ---
col1, col2, col3 = st.columns(3)
with col1:
    L = st.number_input("母料长度 L", value=115.0)
with col2:
    w_input = st.text_input("零件规格 (w_i)", "25, 40, 50, 55, 70")
with col3:
    b_input = st.text_input("需求数量 (b_i)", "50, 36, 24, 8, 30")

# 用于存储迭代过程的列表
if 'iteration_log' not in st.session_state:
    st.session_state.iteration_log = []

if st.button("开始云端求解"):
    st.session_state.iteration_log = [] # 清空之前的日志
    st.write("### 🔄 列生成迭代过程")
    log_area = st.empty() # 用于动态更新日志的区域

    try:
        # 1. 解析数据
        w = [float(x.strip()) for x in w_input.split(",")]
        b = [float(x.strip()) for x in b_input.split(",")]
        n = len(w)

        # 2. 初始化 Gurobi 环境
        if params:
            env = gp.Env(params=params)
        else:
            env = gp.Env()
        
        # --- 3. 初始化受限主问题 (RMP) - 线性松弛 ---
        rmp = gp.Model("RMP", env=env)
        rmp.Params.OutputFlag = 0
        
        # 初始切割模式：单位阵 (每个零件单独切一根)
        patterns = [[(L // w[i] if i == j else 0) for j in range(n)] for i in range(n)]
        
        # 添加初始决策变量 (连续变量，允许小数)
        x = rmp.addVars(n, obj=1.0, vtype=GRB.CONTINUOUS, name="x")
        
        # 添加需求约束
        constrs = rmp.addConstrs((gp.quicksum(x[j] * patterns[j][i] for j in range(n)) >= b[i]) for i in range(n))

        # --- 4. 列生成循环 ---
        iter_count = 0
        while True:
            iter_count += 1
            rmp.optimize()
            
            # 获取对偶值 (Shadow Prices)
            duals = [c.Pi for c in constrs]
            
            # 求解子问题 (背包问题) - 寻找检验数为负的新列
            sub = gp.Model("Sub", env=env)
            sub.Params.OutputFlag = 0
            y = sub.addVars(n, vtype=GRB.INTEGER, name="y")
            sub.setObjective(gp.quicksum(duals[i] * y[i] for i in range(n)), GRB.MAXIMIZE)
            sub.addConstr(gp.quicksum(w[i] * y[i] for i in range(n)) <= L)
            sub.optimize()
            
            # 检查检验数 z* = 1 - sub.ObjVal
            z_star = 1 - sub.ObjVal
            
            # 记录迭代日志
            log_msg = f"**迭代 {iter_count}**: RMP目标值(小数)={round(rmp.ObjVal, 2)}, 子问题目标值={round(sub.ObjVal, 4)}, 检验数={round(z_star, 4)}"
            st.session_state.iteration_log.append(log_msg)
            
            # 动态更新网页上的日志展示
            with log_area.container():
                for msg in st.session_state.iteration_log:
                    st.markdown(msg)

            # 最优性条件：找不到检验数为负的列
            if z_star >= -1e-6:
                st.success("达到最优性条件，列生成过程结束。")
                break
            
            # 发现新模式，将其添加作为 RMP 的新列
            new_pattern = [int(y[i].X + 0.5) for i in range(n)]
            st.session_state.iteration_log.append(f"&nbsp;&nbsp;&nbsp;&nbsp;💡 发现新切割模式: {new_pattern}")
            patterns.append(new_pattern)
            
            # 向 RMP 添加新列
            new_col = gp.Column(new_pattern, constrs)
            rmp.addVar(obj=1.0, column=new_col, vtype=GRB.CONTINUOUS, name=f"x_{len(patterns)-1}")

        # --- 5. 最终整数规划求解 (MIP) ---
        st.write("---")
        st.write("### 🏆 最终整数切割方案")
        st.info("正在将生成的列汇总，求解整数规划问题...")
        
        # 将 RMP 中的所有决策变量转换为整数变量
        for v in rmp.getVars():
            v.vtype = GRB.INTEGER
        
        rmp.optimize()
        
        if rmp.status == GRB.OPTIMAL:
            st.success(f"✅ MIP 计算完成！总共消耗母料: **{int(rmp.ObjVal + 0.5)}** 根")
            
            # 解析排产清单
            results = []
            for j, v in enumerate(rmp.getVars()):
                if v.X > 0.5: # 变量值大于0，说明使用了该方案
                    patterns_str = " + ".join([f"{count}个[{w[i]}mm]" for i, count in enumerate(patterns[j]) if count > 0])
                    results.append({
                        "方案编号": f"Plan-{j+1}",
                        "切割模式": patterns_str,
                        "所需母料根数": int(v.X + 0.5)
                    })
            
            # 使用表格展示排产单
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
            
        else:
            st.error("MIP 求解未成功，请检查数据。")

    except Exception as e:
        st.error(f"求解失败：{e}")
        if "No Gurobi license found" in str(e):
            st.warning("请在左侧配置有效的 Gurobi 许可证。")
