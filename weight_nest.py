import pandas as pd 
import numpy as np 
 
# 假设您的资产包结构如下（根据您的描述）

 
# 函数1：计算单个日期的资产权重 
def calculate_asset_weights(date, df_A, portfolios):
    """
    计算指定日期所有资产的权重 
    
    参数:
    date: 日期 (必须存在于df_A的索引中)
    df_A: 资产包权重DataFrame (index=日期, columns=资产包名称)
    portfolios: 资产包结构字典 
    
    返回:
    Series: 资产权重 (index=资产名称, values=权重)
    """
    # 获取该日期的资产包权重 
    portfolio_weights = df_A.loc[date] 
    
    # 初始化资产权重字典 
    asset_weights = {}
    
    # 遍历每个资产包
    for portfolio, weight in portfolio_weights.items(): 
        # 跳过权重为0的资产包
        if weight == 0:
            continue 
            
        # 获取该资产包的内部资产结构 
        assets = portfolios.get(portfolio,  {})
        
        # 计算每个资产的实际权重并累加
        for asset, internal_weight in assets.items(): 
            asset_weights[asset] = asset_weights.get(asset,  0) + weight * internal_weight 
    
    return pd.Series(asset_weights)
 
# 函数2：生成所有资产的权重DataFrame 
def generate_asset_dataframe(df_A, portfolios):
    """
    生成所有资产的权重DataFrame 
    
    参数:
    df_A: 资产包权重DataFrame (index=日期, columns=资产包名称)
    portfolios: 资产包结构字典 
    
    返回:
    DataFrame: 资产权重DataFrame (index=日期, columns=资产名称)
    """
    # 获取所有唯一资产名称 
    all_assets = set()
    for assets in portfolios.values(): 
        all_assets.update(assets.keys()) 
    all_assets = sorted(all_assets)  # 排序以确保列顺序一致 
    
    # 创建结果DataFrame 
    df_B = pd.DataFrame(index=df_A.index,  columns=all_assets)
    df_B = df_B.fillna(0)   # 初始化为0
    
    # 遍历每个日期并计算资产权重 
    for date in df_A.index: 
        asset_weights = calculate_asset_weights(date, df_A, portfolios)
        df_B.loc[date,  asset_weights.index]  = asset_weights.values  
    
    return df_B 

if __name__ == "__main__":
    asset_portfolios = {
        "portfolio_1": {
            "asset_A": 0.4,
            "asset_B": 0.6 
        },
        "portfolio_2": {
            "asset_B": 0.3,
            "asset_C": 0.7 
        },
        "portfolio_3": {
            "asset_A": 0.2,
            "asset_D": 0.5,
            "asset_E": 0.3 
        },
        "portfolio_4": {
            "asset_C": 0.4,
            "asset_D": 0.6 
        }
    }

    # 示例数据 - 资产包权重DataFrame 
    dates = pd.date_range('2023-01-01',  periods=3)
    df_A = pd.DataFrame({
        'portfolio_1': [0.3, 0.2, 0.4],
        'portfolio_2': [0.4, 0.3, 0.1],
        'portfolio_3': [0.2, 0.4, 0.3],
        'portfolio_4': [0.1, 0.1, 0.2]
    }, index=dates)
    
    print("资产包权重DataFrame (df_A):")
    print(df_A)
    print("\n")
    
    # 计算单个日期的资产权重 
    date = dates[0]
    asset_weights = calculate_asset_weights(date, df_A, asset_portfolios)
    print(f"在 {date} 的资产权重:")
    print(asset_weights)
    print("\n")
    
    # 生成完整的资产权重DataFrame 
    df_B = generate_asset_dataframe(df_A, asset_portfolios)
    print("资产权重DataFrame (df_B):")
    print(df_B)