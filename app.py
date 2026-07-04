import matplotlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import shutil

shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)
plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC', 'WenQuanYi Zen Hei', 'WenQuanYi Micro Hei','DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="光伏+储能优化模拟器", layout="wide")

def load_data(uploaded_file):
    """加载数据"""
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file, sep=",")
    else:
        df = pd.read_csv(uploaded_file, sep="\t")
    df = df[['time', 'Generation', 'Consumption']]
    df['time'] = pd.to_datetime('2023-01-01') + pd.to_timedelta(range(len(df)), unit='h')
    df.set_index('time', inplace=True)
    return df


def simulate_system(df, pv_scale, battery_capacity, 
                   battery_efficiency=0.95, soc_min=0.1, soc_max=0.9, 
                   return_details=False):
    """模拟系统（含充放电状态标记）"""
    generation = df['Generation'].values * pv_scale
    consumption = df['Consumption'].values
    n = len(generation)
    
    battery_soc = np.zeros(n)
    battery_soc[0] = battery_capacity * 0.5
    
    self_consumed = np.zeros(n)
    curtailed = np.zeros(n)
    grid_import = np.zeros(n)
    charge_discharge_flag = np.full(n, '闲置', dtype=object)
    
    for t in range(n):
        gen = generation[t]
        load = consumption[t]
        
        direct_use = min(gen, load)
        self_consumed[t] = direct_use
        remaining_gen = gen - direct_use
        remaining_load = load - direct_use
        
        current_soc = battery_soc[t-1] if t > 0 else battery_soc[0]
        
        if remaining_gen > 0:
            # 充电
            charge_possible = min(remaining_gen, (battery_capacity * soc_max - current_soc) / battery_efficiency)
            battery_soc[t] = current_soc + charge_possible * battery_efficiency
            curtailed[t] = remaining_gen - charge_possible
            if charge_possible > 50:
                charge_discharge_flag[t] = '充电'
        else:
            # 放电
            discharge_possible = min(remaining_load, (current_soc - battery_capacity * soc_min) * battery_efficiency)
            self_consumed[t] += discharge_possible
            battery_soc[t] = current_soc - discharge_possible / battery_efficiency
            grid_import[t] = remaining_load - discharge_possible
            if discharge_possible > 50:
                charge_discharge_flag[t] = '放电'
        
        battery_soc[t] = np.clip(battery_soc[t], battery_capacity * soc_min, battery_capacity * soc_max)
    
    total_gen = np.sum(generation)
    total_load = np.sum(consumption)
    total_self = np.sum(self_consumed)
    total_curtailed = np.sum(curtailed)
    
    result = {
        'pv_scale': pv_scale,
        'battery_capacity': battery_capacity,
        'coverage_rate': total_self / total_load * 100,
        'self_consumption_rate': total_self / total_gen * 100 if total_gen > 0 else 0,
        'curtailment_rate': total_curtailed / total_gen * 100 if total_gen > 0 else 0,
    }
    
    if return_details:
        result['battery_soc'] = battery_soc
        result['self_consumed'] = self_consumed
        result['curtailed'] = curtailed
        result['grid_import'] = grid_import
        result['generation'] = generation
        result['consumption'] = consumption
        result['charge_discharge_flag'] = charge_discharge_flag
    
    return result


def calculate_daily_utilization(battery_soc, battery_capacity):
    n_days = len(battery_soc) // 24
    daily_soc = battery_soc.reshape(n_days, 24)
    daily_max = daily_soc.max(axis=1)
    daily_min = daily_soc.min(axis=1)
    daily_util = (daily_max - daily_min) / battery_capacity * 100
    return daily_util.mean()


def calculate_daily_cycles(battery_soc, battery_capacity):
    """统计每日有效充放电循环次数"""
    n_days = len(battery_soc) // 24
    daily_cycles = []
    for d in range(n_days):
        soc_day = battery_soc[d*24:(d+1)*24]
        diff = np.diff(soc_day)
        significant = battery_capacity * 0.08
        charge_events = np.sum(diff > significant)
        discharge_events = np.sum(diff < -significant)
        cycles = min(charge_events, discharge_events)
        if cycles == 0:
            cycles = max(charge_events, discharge_events) * 0.5
        daily_cycles.append(cycles)
    return daily_cycles, np.mean(daily_cycles)


def optimize_parameters(df, pv_min, pv_max, pv_steps, bat_min, bat_max, bat_steps):
    """网格搜索优化"""
    pv_scales = np.linspace(pv_min, pv_max, pv_steps)
    battery_sizes = np.linspace(bat_min, bat_max, bat_steps)
    
    best_score = -np.inf
    best_config = None
    results = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(pv_scales) * max(1, len(battery_sizes) - 1)
    current = 0
    
    for i, pv in enumerate(pv_scales):
        for bat in battery_sizes:
            if bat < 1000: continue
            res = simulate_system(df, pv, bat)
            temp_sim = simulate_system(df, pv, bat, return_details=True)
            daily_util = calculate_daily_utilization(temp_sim['battery_soc'], bat)
            _, avg_cycles = calculate_daily_cycles(temp_sim['battery_soc'], bat)
            
            score = (res['coverage_rate'] * 0.65 + 
                    res['self_consumption_rate'] * 0.25 + 
                    daily_util * 0.10)
            
            res['score'] = score
            res['daily_utilization'] = daily_util
            res['avg_daily_cycles'] = avg_cycles
            results.append(res)
            
            if score > best_score:
                best_score = score
                best_config = res.copy()
            
            current += 1
            if total > 0:
                progress_bar.progress(min(current / total, 1.0))
            status_text.text(f"正在优化... PV: {pv:.2f} | 电池: {bat/1000:.1f} MWh")
    
    progress_bar.empty()
    status_text.empty()
    
    return best_config, pd.DataFrame(results)


def generate_hourly_table(df, sim_result):
    """生成每小时表格"""
    battery_soc = sim_result['battery_soc']
    generation = sim_result['generation']
    consumption = sim_result['consumption']
    self_consumed = sim_result['self_consumed']
    curtailed = sim_result['curtailed']
    grid_import = sim_result['grid_import']
    flag = sim_result['charge_discharge_flag']
    
    dates = df.index
    hourly_df = pd.DataFrame({
        '日期': dates.date,
        '小时': dates.hour,
        '光伏发电(kW)': np.round(generation, 2),
        '负荷(kW)': np.round(consumption, 2),
        '储能SOC(kWh)': np.round(battery_soc, 2),
        '自发自用(kW)': np.round(self_consumed, 2),
        '弃光(kW)': np.round(curtailed, 2),
        '电网购电(kW)': np.round(grid_import, 2),
        'SOC(%)': np.round(battery_soc / sim_result['battery_capacity'] * 100, 2),
        '充放电状态': flag,
    })
    return hourly_df


def plot_daily_pv_trend(df, pv_scale):
    daily_gen = df['Generation'].resample('D').sum() * pv_scale
    days = range(1, len(daily_gen) + 1)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(days, daily_gen / 1000, 'b-', linewidth=1.8, label='每日光伏发电量')
    ax.axhline(y=daily_gen.mean()/1000, color='red', linestyle='--', 
               label=f'平均每日发电: {daily_gen.mean()/1000:.1f} MWh')
    ax.set_title(f'全年光伏每日发电趋势 (装机倍数: {pv_scale:.2f})')
    ax.set_xlabel('天数')
    ax.set_ylabel('发电量 (MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


def plot_daily_utilization(sim_result, battery_capacity):
    battery_soc = sim_result['battery_soc']
    n_days = len(battery_soc) // 24
    daily_soc = battery_soc.reshape(n_days, 24)
    
    daily_max = daily_soc.max(axis=1)
    daily_min = daily_soc.min(axis=1)
    daily_avg = daily_soc.mean(axis=1)
    daily_utilization = (daily_max - daily_min) / battery_capacity * 100
    
    days = range(1, n_days + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    
    axs[0].plot(days, daily_max/1000, 'b-', label='每日最高SOC')
    axs[0].plot(days, daily_min/1000, 'r-', label='每日最低SOC')
    axs[0].plot(days, daily_avg/1000, 'g--', linewidth=2, label='每日平均SOC')
    axs[0].set_ylabel('储能电量 (MWh)')
    axs[0].set_title('每日储能SOC变化范围')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    axs[1].plot(days, daily_utilization, 'purple', linewidth=2, label='每日利用率 (%)')
    axs[1].axhline(y=daily_utilization.mean(), color='red', linestyle='--', 
                   label=f'平均利用率: {daily_utilization.mean():.1f}%')
    axs[1].set_xlabel('天数')
    axs[1].set_ylabel('利用率 (%)')
    axs[1].set_title('每日储能利用率')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    plt.suptitle(f'全年每日储能分析 (PV倍数: {sim_result["pv_scale"]:.2f})', fontsize=14)
    plt.tight_layout()
    return fig


def plot_soc_with_events(sim_result, days_to_show=30):
    soc = sim_result['battery_soc']
    flag = sim_result['charge_discharge_flag']
    hours = range(days_to_show * 24)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(hours, soc[:len(hours)] / 1000, 'b-', linewidth=2, label='储能SOC (MWh)')
    
    charge_idx = np.where(flag[:len(hours)] == '充电')[0]
    discharge_idx = np.where(flag[:len(hours)] == '放电')[0]
    
    ax.scatter(charge_idx, soc[charge_idx]/1000, color='green', s=60, marker='^', label='充电事件')
    ax.scatter(discharge_idx, soc[discharge_idx]/1000, color='red', s=60, marker='v', label='放电事件')
    
    ax.set_title(f'前{days_to_show}天 SOC曲线与充放电事件标记')
    ax.set_xlabel('小时')
    ax.set_ylabel('储能电量 (MWh)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig


# ====================== Streamlit 主界面 ======================

st.title("光伏 + 储能 系统优化模拟器")
st.markdown("**基于8760小时数据进行参数优化与详细模拟**")

# 侧边栏
with st.sidebar:
    st.header("📁 数据上传")
    uploaded_file = st.file_uploader("上传负荷与发电数据 (CSV/TSV)", type=["csv", "txt", "xls"])
    
    st.header("⚙️ 优化参数范围")
    pv_min = st.slider("PV装机倍数 最小", 0.5, 1.5, 0.6, 0.05)
    pv_max = st.slider("PV装机倍数 最大", 1.5, 3.0, 2.0, 0.05)
    pv_steps = st.slider("PV搜索步数", 5, 20, 11)
    
    bat_min = st.slider("储能容量 最小 (MWh)", 10, 100, 20) * 1000
    bat_max = st.slider("储能容量 最大 (MWh)", 100, 500, 250) * 1000
    bat_steps = st.slider("储能搜索步数", 5, 15, 9)
    
    run_button = st.button("开始优化计算", type="primary")

if uploaded_file is None:
    st.info("👆 请先上传数据文件（需包含 `time`、`Generation`、`Consumption` 三列）")
    st.stop()

if run_button:
    with st.spinner("正在加载数据并进行优化计算..."):
        df = load_data(uploaded_file)
        st.success(f"✅ 数据加载成功！共 {len(df):,} 小时数据")
        
        best_config, results_df = optimize_parameters(
            df, pv_min, pv_max, pv_steps, bat_min, bat_max, bat_steps
        )
        
        detailed_sim = simulate_system(
            df, best_config['pv_scale'], best_config['battery_capacity'], return_details=True
        )
        
        # 结果展示
        st.success("优化计算完成！")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("推荐PV装机倍数", f"{best_config['pv_scale']:.2f} 倍")
        with col2:
            st.metric("推荐储能容量", f"{best_config['battery_capacity']/1000:.1f} MWh")
        with col3:
            st.metric("负荷覆盖率", f"{best_config['coverage_rate']:.1f}%")
        with col4:
            st.metric("综合评分", f"{best_config['score']:.2f}")
        
        st.subheader("📊 优化参数汇总 (Top 10)")
        summary = results_df[['pv_scale', 'battery_capacity', 'coverage_rate', 
                            'self_consumption_rate', 'daily_utilization', 
                            'avg_daily_cycles', 'score']].copy()
        summary['battery_capacity_mwh'] = summary['battery_capacity'] / 1000
        summary = summary.round(2).sort_values('score', ascending=False)
        st.dataframe(summary.head(10), use_container_width=True)
        
        csv = summary.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载优化汇总表", csv, "optimization_summary.csv", "text/csv")
        
        # 小时级明细
        hourly_df = generate_hourly_table(df, detailed_sim)
        st.subheader("⏰ 8760小时模拟明细 (前200行预览)")
        st.dataframe(hourly_df.head(200), use_container_width=True)
        
        csv_hourly = hourly_df.to_csv(index=False).encode('utf-8-sig')
        st.download_button("📥 下载完整8760小时明细表", csv_hourly, "hourly_simulation.csv", "text/csv")
        
        # 图表
        st.subheader("📈 可视化分析")
        tab1, tab2, tab3 = st.tabs(["每日光伏发电趋势", "每日储能利用率", "SOC曲线与充放电事件"])
        
        with tab1:
            fig1 = plot_daily_pv_trend(df, best_config['pv_scale'])
            st.pyplot(fig1)
        
        with tab2:
            fig2 = plot_daily_utilization(detailed_sim, best_config['battery_capacity'])
            st.pyplot(fig2)
        
        with tab3:
            fig3 = plot_soc_with_events(detailed_sim, days_to_show=30)
            st.pyplot(fig3)

else:
    st.info("👈 在左侧侧边栏调整参数后，点击 **开始优化计算**")
