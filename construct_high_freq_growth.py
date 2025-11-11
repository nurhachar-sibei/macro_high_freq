"""
构建高频Growth宏观因子
方法：
1. 以四个资产的对数同比序列为自变量
2. 对原始Growth因子进行多变量领先滞后回归，确定领先期及各资产回归系数
3. 将归一化回归系数为权重，对自变量的环比序列进行加权，从而得到高频因子的环比序列
4. 作图，将高频因子的净值曲线和Growth宏观因子曲线做对比
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 读取数据
print("=" * 60)
print("Step 1: 读取数据")
print("=" * 60)

# 读取月频宏观因子
macro = pd.read_csv('macro.csv', index_col=0, parse_dates=True)
print(f"宏观因子数据: {macro.shape}")
print(macro.head())

# 读取周频资产价格
grow_data = pd.read_csv('grow_data.csv', index_col=0, parse_dates=True)
print(f"\n周频资产价格数据: {grow_data.shape}")
print(grow_data.head())

# 资产名称
asset_names = grow_data.columns.tolist()
print(f"\n资产列表: {asset_names}")

# 2. 计算四个资产的对数同比序列（作为自变量）
print("\n" + "=" * 60)
print("Step 2: 计算资产的对数同比序列")
print("=" * 60)

# 对数收益率
log_price = np.log(grow_data)


# 同比序列 (52周前，约1年)
log_yoy = log_price - log_price.shift(52)
log_yoy = log_yoy.dropna()
print(f"对数同比序列: {log_yoy.shape}")
print(log_yoy.head())

# 3. 将周频数据转换为月频，与Growth因子对齐
print("\n" + "=" * 60)
print("Step 3: 周频数据转月频，与Growth因子对齐")
print("=" * 60)

# 取月末数据
log_yoy_monthly = log_yoy.resample('M').last()
print(f"月频对数同比序列: {log_yoy_monthly.shape}")

# 对齐数据
# 找到共同的时间区间
common_dates = macro.index.intersection(log_yoy_monthly.index)
print(f"共同时间区间: {len(common_dates)} 个月")
print(f"时间范围: {common_dates[0]} 到 {common_dates[-1]}")

# 对齐数据
growth_monthly = macro.loc[common_dates, 'Growth']
X_monthly = log_yoy_monthly.loc[common_dates, :]

# 4. 进行多变量领先滞后回归，确定最优领先期
print("\n" + "=" * 60)
print("Step 4: 多变量领先滞后回归，确定最优领先期")
print("=" * 60)

# 测试领先期范围（-6到6个月）
lead_lags = range(-6, 7)
best_score = -np.inf
best_lead_lag = 0
best_model = None
scores = []

for lead_lag in lead_lags:
    # 调整Y的时间
    if lead_lag >= 0:
        y_shifted = growth_monthly.iloc[lead_lag:]
        X_shifted = X_monthly.iloc[:-lead_lag] if lead_lag > 0 else X_monthly
    else:
        y_shifted = growth_monthly.iloc[:lead_lag]
        X_shifted = X_monthly.iloc[-lead_lag:]
    
    # 对齐索引
    common_idx = y_shifted.index.intersection(X_shifted.index)
    if len(common_idx) < 10:  # 至少需要10个样本
        continue
    
    y_aligned = y_shifted.loc[common_idx]
    X_aligned = X_shifted.loc[common_idx]
    
    # 线性回归
    model = LinearRegression()
    model.fit(X_aligned, y_aligned)
    score = model.score(X_aligned, y_aligned)
    scores.append(score)
    
    print(f"领先期={lead_lag:2d}月, R²={score:.4f}")
    
    if score > best_score:
        best_score = score
        best_lead_lag = lead_lag
        best_model = model

print(f"\n最优领先期: {best_lead_lag}月, R²={best_score:.4f}")
print(f"回归系数: {dict(zip(asset_names, best_model.coef_))}")

# 5. 归一化回归系数作为权重
print("\n" + "=" * 60)
print("Step 5: 归一化回归系数作为权重")
print("=" * 60)

# 取绝对值后归一化（使权重和为1）
weights_raw = np.abs(best_model.coef_)
weights = weights_raw / weights_raw.sum()

print("归一化权重:")
for name, weight in zip(asset_names, weights):
    print(f"  {name}: {weight:.4f}")

# 6. 计算周频资产的环比序列
print("\n" + "=" * 60)
print("Step 6: 计算周频资产的环比序列")
print("=" * 60)

# 环比 = log(P_t) - log(P_{t-1})
log_mom = log_price.diff()
log_mom = log_mom.dropna()
print(f"对数环比序列: {log_mom.shape}")
print(log_mom.head())

# 7. 加权得到高频Growth因子的环比序列
print("\n" + "=" * 60)
print("Step 7: 加权得到高频Growth因子环比序列")
print("=" * 60)

# 加权
high_freq_growth_mom = (log_mom * weights).sum(axis=1)
print(f"高频Growth环比序列: {high_freq_growth_mom.shape}")
print(high_freq_growth_mom.head())

# 8. 累积环比得到高频Growth因子水平值
print("\n" + "=" * 60)
print("Step 8: 累积环比得到高频Growth因子水平值")
print("=" * 60)

# 累积求和（从0开始）
high_freq_growth = high_freq_growth_mom.cumsum()
print(f"高频Growth因子: {high_freq_growth.shape}")

# 9. 将Growth月频因子也转换为净值曲线（累积）
print("\n" + "=" * 60)
print("Step 9: 转换Growth月频因子为净值曲线")
print("=" * 60)

# Growth因子的环比
growth_monthly_sorted = growth_monthly.sort_index()
growth_mom = growth_monthly_sorted.diff()
growth_cumsum = growth_mom.cumsum()

# 为了便于比较，将两个序列标准化到同一起点
# 找到高频因子与月频因子的共同时间区间
common_dates_compare = growth_cumsum.index.intersection(high_freq_growth.index)
print(f"用于对比的共同时间区间: {len(common_dates_compare)} 个点")

if len(common_dates_compare) > 0:
    # 标准化到同一起点（第一个值为0）
    growth_norm = growth_cumsum.loc[common_dates_compare] - growth_cumsum.loc[common_dates_compare].iloc[0]
    high_freq_norm = high_freq_growth.loc[common_dates_compare] - high_freq_growth.loc[common_dates_compare].iloc[0]
    
    # 10. 作图对比
    print("\n" + "=" * 60)
    print("Step 10: 绘图对比")
    print("=" * 60)
    
    plt.figure(figsize=(14, 7))
    
    # 绘制高频因子（周频）
    plt.plot(high_freq_growth.index, 
             high_freq_growth - high_freq_growth.iloc[0], 
             label='高频Growth因子（周频）', 
             linewidth=1, 
             alpha=0.8)
    
    # 绘制月频因子
    plt.plot(growth_cumsum.index, 
             growth_cumsum - growth_cumsum.iloc[0], 
             label='原始Growth因子（月频）', 
             linewidth=2, 
             marker='o', 
             markersize=4,
             alpha=0.7)
    
    plt.title('高频Growth因子 vs 原始Growth因子对比', fontsize=16)
    plt.xlabel('日期', fontsize=12)
    plt.ylabel('累积值', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('high_freq_growth_comparison.png', dpi=300, bbox_inches='tight')
    print("图表已保存为: high_freq_growth_comparison.png")
    
    # 额外绘制相关性散点图
    # 将高频因子重采样到月频
    high_freq_monthly = high_freq_growth.resample('M').last()
    common_for_corr = growth_monthly.index.intersection(high_freq_monthly.index)
    
    if len(common_for_corr) > 0:
        plt.figure(figsize=(10, 7))
        plt.scatter(growth_monthly.loc[common_for_corr], 
                   high_freq_monthly.loc[common_for_corr],
                   alpha=0.6)
        
        # 计算相关系数
        corr = np.corrcoef(growth_monthly.loc[common_for_corr], 
                          high_freq_monthly.loc[common_for_corr])[0, 1]
        
        plt.title(f'高频Growth因子 vs 原始Growth因子散点图\n相关系数: {corr:.4f}', 
                 fontsize=16)
        plt.xlabel('原始Growth因子（月频）', fontsize=12)
        plt.ylabel('高频Growth因子（月频重采样）', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('high_freq_growth_correlation.png', dpi=300, bbox_inches='tight')
        print("相关性图表已保存为: high_freq_growth_correlation.png")
        print(f"\n相关系数: {corr:.4f}")

# 11. 保存高频Growth因子
print("\n" + "=" * 60)
print("Step 11: 保存高频Growth因子")
print("=" * 60)

# 保存为CSV
high_freq_growth.to_csv('high_freq_growth.csv', header=['High_Freq_Growth'])
print("高频Growth因子已保存为: high_freq_growth.csv")

# 同时保存包含权重信息的文件
weights_df = pd.DataFrame({
    'Asset': asset_names,
    'Weight': weights,
    'Raw_Coefficient': best_model.coef_
})
weights_df.to_csv('growth_weights.csv', index=False)
print("权重信息已保存为: growth_weights.csv")

print("\n" + "=" * 60)
print("完成！")
print("=" * 60)
print(f"最优领先期: {best_lead_lag}月")
print(f"R²: {best_score:.4f}")
print(f"高频因子数据点数: {len(high_freq_growth)}")
