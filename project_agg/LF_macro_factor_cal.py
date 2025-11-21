'''
用于计算低频宏观因子（月频率）
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.filters.hp_filter import hpfilter

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LF_macro_factor_cal:
    def __init__(self, factor_plot=False, dataframe=None):
        """
        初始化低频宏观因子计算类
        
        参数:
        factor_plot: bool, 是否绘制因子图表
        dataframe: pd.DataFrame, 输入的宏观经济数据DataFrame
        
        注意:
        HP滤波逻辑已固定：
        - 2012年及以前：对整个序列进行HP滤波（样本量较少）
        - 2013年及以后：每个样本只使用它之前的时间序列进行HP滤波（避免未来信息泄露）
        """
        self.factor_plot = factor_plot
        self.dataframe = dataframe.copy() if dataframe is not None else None
        self.factors = {}  # 存储计算出的因子
        
    def _apply_hp_filter(self, series, lambda_param=1):
        """
        应用HP滤波，避免未来信息泄露
        
        逻辑：
        - 2012年及以前：由于样本量较少，对整个序列进行HP滤波
        - 2013年及以后：每个样本只使用它之前的时间序列进行HP滤波（滚动窗口）
        
        参数:
        series: pd.Series, 需要滤波的序列
        lambda_param: float, HP滤波的lambda参数，默认1（月频数据）
        
        返回:
        trend: pd.Series, 趋势项（保持原Series的索引和长度）
        """
        # 获取非空数据的索引
        valid_mask = series.notna()
        valid_series = series[valid_mask]
        
        if len(valid_series) == 0:
            return pd.Series(index=series.index, dtype=float)
        
        # 定义2012年和2013年的分界点
        cutoff_date = pd.Timestamp('2020-12-31')
        
        # 创建结果Series
        full_trend = pd.Series(index=series.index, dtype=float)
        
        # 分离2012年及以前和2013年及以后的数据
        before_2013 = valid_series[valid_series.index <= cutoff_date]
        after_2012 = valid_series[valid_series.index > cutoff_date]
        
        # 处理2012年及以前的数据：对整个序列进行HP滤波
        if len(before_2013) > 0:
            # HP滤波至少需要3个数据点
            if len(before_2013) >= 3:
                cycle, trend = hpfilter(before_2013.values, lamb=lambda_param)
                full_trend.loc[before_2013.index] = trend
            else:
                # 如果数据点太少，直接使用原值
                full_trend.loc[before_2013.index] = before_2013.values
        
        # 处理2013年及以后的数据：滚动窗口HP滤波
        if len(after_2012) > 0:
            # 对每个时间点，只使用它之前的数据进行HP滤波（避免未来信息泄露）
            for date, value in after_2012.items():
                # 获取该时间点之前的所有数据（包括2012年及以前的数据）
                historical_data = valid_series[valid_series.index <= date]
                
                # HP滤波至少需要3个数据点
                if len(historical_data) >= 3:
                    try:
                        cycle, trend = hpfilter(historical_data.values, lamb=lambda_param)
                        # 只取最后一个值（当前时间点的平滑值）
                        full_trend.loc[date] = trend[-1]
                    except Exception:
                        # 如果HP滤波失败，使用原值
                        full_trend.loc[date] = value
                else:
                    # 如果数据点太少，使用原值
                    full_trend.loc[date] = value
        
        return full_trend
    
    def _plot_factor(self, factor_series, factor_name, save_path=None):
        """
        绘制因子图表
        
        参数:
        factor_series: pd.Series, 因子序列
        factor_name: str, 因子名称
        save_path: str, 保存路径，如果为None则使用默认路径
        """
        if not self.factor_plot:
            return
        
        plt.figure(figsize=(12, 6))
        plt.plot(factor_series.index, factor_series.values, linewidth=1.5, alpha=0.8)
        plt.title(f'{factor_name}因子', fontsize=14, fontweight='bold')
        plt.xlabel('日期', fontsize=12)
        plt.ylabel('因子值', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path is None:
            save_path = f'{factor_name}_factor.png'
        
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f'{factor_name}因子图表已保存至: {save_path}')
    
    def cal_growth_factor(self, start_date='2006-12-31'):
        """
        计算增长因子
        
        增长因子使用以下指标：
        - delta_M0017126: PMI同比差分
        - M0000273: 固定资产投资额完成额同比
        - M0001428: 社会消费品零售总额同比（当月同比）
        - M0000605: 进出口金额同比（当月同比）
        
        根据波动率进行加权构造
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 计算PMI同比差分
        self.dataframe['delta_M0017126'] = self.dataframe['M0017126'].diff()
        
        # 提取增长因子相关数据
        growth_data = self.dataframe[['delta_M0017126', 'M0000273', 'M0001428', 'M0000605']].loc[start_date:].copy()
        
        # 处理一月二月数据合并导致的缺失数据
        growth_data.loc[(growth_data.index.month == 1), ['M0000273', 'M0001428']] = np.nan
        growth_data = growth_data.ffill()
        
        # 应用HP滤波
        lambda_param = 1  # 月频数据
        for column in growth_data.columns:
            series = growth_data[column]
            trend = self._apply_hp_filter(series, lambda_param=lambda_param)
            growth_data[column] = trend
        
        # 计算权重（根据波动率）
        weight = (growth_data.std()) / (growth_data.std()).sum()
        
        # 计算增长因子
        growth_factor = ((weight * growth_data).sum(axis=1) / 100 / 2)
        
        self.factors['Growth'] = growth_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(growth_factor, 'Growth')
        
        return growth_factor
    
    def cal_inflation_factor(self, start_date='2006-12-31'):
        """
        计算通胀因子
        
        通胀因子使用以下指标：
        - M0000612: CPI同比（当月同比）
        - M0001227: PPI同比（当月同比）
        
        按波动率倒数进行加权构造
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 提取通胀因子相关数据
        inflation_data = self.dataframe[['M0000612', 'M0001227']].loc[start_date:].copy()
        
        # 应用HP滤波
        lambda_param = 1  # 月频数据
        for column in inflation_data.columns:
            series = inflation_data[column]
            trend = self._apply_hp_filter(series, lambda_param=lambda_param)
            inflation_data[column] = trend
        
        # 计算权重（根据波动率倒数）
        weight = (1 / inflation_data.std()) / (1 / inflation_data.std()).sum()
        
        # 计算通胀因子
        inflation_factor = ((weight * inflation_data).sum(axis=1) / 100)
        
        self.factors['Inflation'] = inflation_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(inflation_factor, 'Inflation')
        
        return inflation_factor
    
    def cal_interest_factor(self, start_date='2006-12-31'):
        """
        计算利率因子
        
        利率因子使用10年期国债收益率(S0059749)表示
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 提取利率因子数据
        interest_factor = self.dataframe['S0059749'].loc[start_date:] / 100
        
        self.factors['Interest'] = interest_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(interest_factor, 'Interest')
        
        return interest_factor
    
    def cal_credit_factor(self, start_date='2006-12-30'):
        """
        计算信用因子
        
        信用因子使用3年期AA中短期票据收益率(S0059717)和3年期国开债收益率(M1004265)的差来刻画信用利差的变化
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 提取信用因子相关数据
        credit_data = self.dataframe[['S0059717', 'M1004265']].copy()
        
        # 处理缺失数据（线性插值）
        credit_data.loc['2006-12-23':"2008-04-22", 'S0059717'] = np.nan
        if len(credit_data.loc['2006-11-30':'2008-04-30']) > 0:
            linear_value = np.linspace(
                credit_data.loc['2006-11-30', 'S0059717'],
                credit_data.loc['2008-04-30', 'S0059717'],
                len(credit_data.loc['2006-11-30':'2008-04-30'])
            )
            credit_data.loc['2006-12-23':"2008-04-22", 'S0059717'] = linear_value[1:-1]
        
        # 计算信用因子（利差）
        credit_factor = (credit_data['S0059717'] - credit_data['M1004265']).loc[start_date:]
        
        self.factors['Credit'] = credit_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(credit_factor, 'Credit')
        
        return credit_factor
    
    def cal_exchange_factor(self, start_date='2006-12-31'):
        """
        计算汇率因子
        
        汇率因子使用美元指数(M0000271)来表示
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 提取汇率因子数据
        exchange_factor = self.dataframe['M0000271'].loc[start_date:]
        
        self.factors['Exchange'] = exchange_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(exchange_factor, 'Exchange')
        
        return exchange_factor
    
    def cal_liquidity_factor(self, start_date='2006-12-31'):
        """
        计算流动性因子
        
        流动性因子使用M2同比(M0001385)和社融存量同比(M5525763)的差来构造
        """
        if self.dataframe is None:
            raise ValueError("请先传入dataframe数据")
        
        # 提取流动性因子相关数据
        liquidity_data = self.dataframe[['M0001385', 'M5525763']].loc[start_date:].copy()
        
        # 应用HP滤波
        lambda_param = 1  # 月频数据
        for column in liquidity_data.columns:
            series = liquidity_data[column]
            trend = self._apply_hp_filter(series, lambda_param=lambda_param)
            liquidity_data[column] = trend
        
        # 计算流动性因子（差值）
        liquidity_factor = (liquidity_data['M0001385'] - liquidity_data['M5525763']) / 100
        
        self.factors['Liquidity'] = liquidity_factor
        
        # 绘图
        if self.factor_plot:
            self._plot_factor(liquidity_factor, 'Liquidity')
        
        return liquidity_factor
    
    def cal_all_factors(self, start_date='2006-12-31'):
        """
        计算所有因子
        
        参数:
        start_date: str, 开始日期
        
        返回:
        pd.DataFrame, 包含所有因子的DataFrame
        """
        # 计算各个因子
        self.cal_growth_factor(start_date)
        self.cal_inflation_factor(start_date)
        self.cal_interest_factor(start_date)
        self.cal_credit_factor(start_date)
        self.cal_exchange_factor(start_date)
        self.cal_liquidity_factor(start_date)
        
        # 合并所有因子
        macro_factors = pd.concat(list(self.factors.values()), axis=1)
        macro_factors.columns = list(self.factors.keys())
        
        return macro_factors
    
    def save_factors(self, filepath='macro_factors.csv'):
        """
        保存因子到CSV文件
        
        参数:
        filepath: str, 保存路径
        """
        if not self.factors:
            raise ValueError("请先计算因子")
        
        macro_factors = pd.concat(list(self.factors.values()), axis=1)
        macro_factors.columns = list(self.factors.keys())
        macro_factors.to_csv(filepath)
        print(f'因子数据已保存至: {filepath}')

