import pandas as pd
import os 
import matplotlib.pyplot as plt
import numpy as np
from WindPy import w
import cvxpy as cp
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from Util_Fin import easy_manager
from Util_Fin import logger_util
# follow 20230614-国泰海通-大类资产配置量化模型研究系列之四：基于宏观因子的大类资产配置框架

import statsmodels.api  as sm 
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from statsmodels.tsa.filters.hp_filter  import hpfilter

from Util_Fin.PCAanalysis import PCAAnalyzer
from Util_Fin.eval_module import (
    PerformanceEvaluator, 
    RiskAnalyzer, 
    PeriodAnalyzer, 
    ReportGenerator
)


import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
from MFH_config import MFHConfig
from joblib import Parallel, delayed


w.start()

class MFH:
    def __init__(self,config=None):
        self.config = config or MFHConfig
        # 初始化日志
        self.logger = logger_util.setup_logger(
            log_file=self.config.LOG_FILE_PREFIX,
            file_load=self.config.LOG_FILE_PATH
        )
        with easy_manager.EasyManager(database='macro_data_base') as em:
            '''
            ___________________________
            数据库读取
            ___________________________
            '''
            self.logger.info("*"*30+"MFH回测系统"+"*"*60)
            self.logger.info("="*60)
            self.logger.info("开始加载数据库数据，加载日志为-datadeal")
            '''
            加载宏观高频因子数据
            '''
            self.HF_macro_factor = em.load_table("macro_high_freq_data_weekly_pct")
            self.HF_macro_factor.set_index('index',inplace=True)
            self.HF_macro_factor.index = pd.to_datetime(self.HF_macro_factor.index)
            '''
            加载价格数据-日
            '''
            self.price_df_d = em.load_table('daily_asset_price_1')
            self.price_df_d.set_index('index',inplace=True)
            self.price_df_d.index = pd.to_datetime(self.price_df_d.index)
            self.price_df_d.dropna(inplace=True)
            self.ret_D_df = self.price_df_d[self.config.BENCHMARK_ASSET].pct_change().dropna()
            '''
            调整价格数据-周
            '''
            self.price_df = self.price_df_d.resample('W').last()
            self.price_df.dropna(inplace=True)
            '''
            计算周收益率
            '''
            self.ret_W_df = self.price_df.pct_change().dropna().loc[:self.config.END_DATE]
            self.HF_macro_factor,self.ret_W_df = self.index_common(self.HF_macro_factor,self.ret_W_df,print_info=False)

            self.logger.info("="*60)
            self.logger.info("数据库读取完成")
            self.logger.info("="*60)
        
        '''
        benchmark model下的权重以及持仓日信息获取
        '''
        self.bench_chg_position = pd.read_excel(self.config.BENCHMARK_MODEL_POSITION_PATH)
        self.bench_weight_df = pd.read_excel(self.config.BENCHMARK_MODEL_WEIGHT_PATH)
        self.bench_weight_df.set_index("index",inplace=True)
        self.bench_weight_df.index = pd.to_datetime(self.bench_weight_df.index)
        self.bench_position_weight = self.bench_weight_df[self.bench_weight_df.index.isin(self.bench_chg_position['index'])]
        # self.bench_position_weight.set_index("index",inplace=True)
        self.bench_position_weight.index = pd.to_datetime(self.bench_position_weight.index)
        #周频化
        self.bench_weight = self.bench_position_weight.resample("W").last().fillna(method="ffill").dropna()
        self.bench_weight,self.ret_W_df_2 = self.index_common(self.bench_weight,self.ret_W_df,print_info=False)
        self.HF_macro_factor_2,self.ret_W_df_2 = self.index_common(self.HF_macro_factor,self.ret_W_df_2,print_info=False)

        self.logger.info("="*60)
        self.logger.info("策略运行器初始化完成")
        self.logger.info("="*60)






    def index_common(self,df1,df2,print_info = True):
        '''
        计算两个DataFrame的公共索引
        '''
        common_idx = df1.index.intersection(df2.index)
        index_df1_only = df1.index.difference(df2.index)
        index_df2_only = df2.index.difference(df1.index)
        df1 = df1.loc[common_idx]
        df2 = df2.loc[common_idx]
        if print_info:
            print(f"df1中存在但df2中不存在的日期: {index_df1_only}")
            print(f"df2中存在但df1中不存在的日期: {index_df2_only}")
        return df1,df2
    
    def exposure_cal(self,date,ret_df=None,HF=None):
        '''
        计算单个日期各项资产的暴露值
        '''
        # ret_df=ret_df or self.ret_W_df
        # # HF=HF or self.HF_macro_factor
        # HF=HF or self.HF_macro_factor_2
        # print(f"计算{date}的因子暴露")
        exposures_df = pd.DataFrame(index=ret_df.columns, columns=HF.columns)
        # exposures_df.fillna(0,inplace=True)
        HF_macro_factor_date = HF.loc[:date].iloc[-260:]
        ret_W_df_date = ret_df.loc[:date].iloc[-260:]
        R2_dict = {}
        residuals_df = pd.DataFrame(index=ret_W_df_date.index, columns=ret_W_df_date.columns)
        for asset in ret_df.columns:
            R2_list = []
            lookback = 260
            halflife = 52
            weights = np.exp(-np.log(2) * np.arange(lookback)[::-1] / halflife)
            weights /= weights.sum()  # 归一化权重

            # 决定本资产回归时是否包含Credit_HF
            if asset in ["CBA00601_CS", "CBA02001_CS", "000832_CSI"]:
                # 用全部因子
                curr_HF_cols = HF.columns
            else:
                # 不包括Credit_HF
                curr_HF_cols = HF.columns.drop("Credit_HF")

            HF_macro_factor_date_reg = HF_macro_factor_date[curr_HF_cols]
            Y = ret_W_df_date[asset].values
            X = HF_macro_factor_date_reg.values
            X = sm.add_constant(X)
            model = sm.WLS(Y, X, weights=weights).fit()
            # 因为去掉了Credit_HF的情况下，params长度变短，需要和 exposures_df 对齐
            params = model.params[1:]
            exposure_vals = np.full((len(HF.columns),), np.nan)
            residuals_df[asset] = Y-model.predict()
            for i, cname in enumerate(curr_HF_cols):
                exposure_vals[HF.columns.get_loc(cname)] = params[i]
            exposures_df.loc[asset] = exposure_vals
            R2_dict[asset] = model.rsquared
        # exposures_df.fillna(0,inplace=True)
        exposures_df.replace(np.nan,0,inplace=True)
        return exposures_df, R2_dict,residuals_df,ret_W_df_date
    

    def exposure_cal_parallel(self):
        '''
        计算所有日期下的组合暴露值
        '''
        self.logger.info("="*60)
        self.logger.info("计算BENCH_MARK暴露值")
        self.logger.info("="*60)
        asset_bench = self.config.BENCHMARK_ASSET
        self.ret_W_df_1 = self.ret_W_df_2[asset_bench]
        weight_bench = self.bench_weight[asset_bench]


        cal_date_list = self.HF_macro_factor_2.index[52*5:]
        # print(cal_date_list)
        #初始化DataFrame
        bench_exposure = pd.DataFrame(index=cal_date_list, columns=self.HF_macro_factor_2.columns)
        residual_dict = {}
        ret_W_df_cal_dict = {}
        #定义处理单个日期函数
        def process_date(date):
            exposures_df, R2_dict,residuals_df,ret_W_df_date = self.exposure_cal(date,self.ret_W_df_1,self.HF_macro_factor_2)
            return date, [np.matrix(np.array(weight_bench.loc[:date].iloc[-1]))* exposures_df.loc[asset_bench].values , 
                        residuals_df,ret_W_df_date]


        # 并行计算（n_jobs=-1 使用所有CPU核心）
        results = Parallel(n_jobs=-1)(
            delayed(process_date)(date) 
            for date in cal_date_list
        )
        # 填充结果
        for date, values in results:
            bench_exposure.loc[date]  = values[0]
            residual_dict[date] = values[1]
            ret_W_df_cal_dict[date] = values[2]
        self.bench_exposure = bench_exposure
        self.residual_dict = residual_dict
        self.ret_W_df_cal_dict = ret_W_df_cal_dict
    
    def Macro_stat_predict(self):
        '''
        预测下一期的因子暴露值
        ------------------------
        该模块可以随意更换，本处使用EMD分解以后的趋势值做动量进行预测
        '''
        self.logger.info("="*60)
        self.logger.info("计算宏观高频因子趋势值")
        self.logger.info("="*60)
        from PyEMD import EMD
        def EMD_trend(df):
            '''
            对df进行EMD分解，返回趋势值
            '''
            resi_df = pd.DataFrame(index=df.index, columns=df.columns)
            for i in df.columns:
                t = np.array(df.index)
                s = np.array(df[i])

                emd = EMD()
                IMFs = emd(s)
                residue =s-(IMFs[0]+IMFs[1])
                resi_df[i] = residue
            return resi_df
        HF_macro_trend = EMD_trend(self.HF_macro_factor_2)
        HF_macro_predict = HF_macro_trend.diff()
        HF_macro_predict[HF_macro_predict>=0] = 1
        HF_macro_predict[HF_macro_predict<0] = -1
        self.HF_macro_predict = HF_macro_predict
        self.HF_macro_trend = HF_macro_trend

    def calculate_dynamic_weight(self, hold_df, init_weight, init_date):
        """
        计算因价格变动产生的动态权重
        
        Parameters:
        -----------
        hold_df : pd.DataFrame
            持有期内所有资产的收益率表格
        init_weight : np.matrix
            第一天所持有的权重(n×1矩阵)
        init_date : pd.Timestamp
            第一天所对应的日期
            
        Returns:
        --------
        weights_df : pd.DataFrame
            动态权重时间序列
        """
        weights_df = pd.DataFrame(
            index=hold_df.index, 
            columns=hold_df.columns
        ).fillna(0)
        
        # 设置初始权重
        weights_df.loc[init_date] = init_weight.T.tolist()[0]
        prev_weights = weights_df.loc[init_date].values
        
        # 逐日更新权重
        for i in range(1, len(weights_df)):
            daily_ret = hold_df.iloc[i].values
            # 计算组合总收益率
            port_return = np.dot(prev_weights, daily_ret)
            # 计算新权重
            new_weights = prev_weights * (1 + daily_ret) / (1 + port_return)
            weights_df.iloc[i] = new_weights
            prev_weights = new_weights
        
        return weights_df



    def MFH_optimal(self):
        self.sigma_beta = self.bench_exposure.std()
        freq = self.config.CHG_FREQ
        self.logger.info("="*60)
        self.logger.info("计算MFH最优权重")
        self.logger.info("="*60)
        HF_macro_predict_1 = self.HF_macro_trend.resample(freq).mean().diff()
        HF_macro_predict_1[HF_macro_predict_1>=0] = 1
        HF_macro_predict_1[HF_macro_predict_1<0] = -1
        bench_position_weight = self.bench_position_weight.loc[self.bench_exposure.index[1]:]
        delta_w_series = pd.DataFrame(index=bench_position_weight.index,columns=bench_position_weight.columns)
        w_series = pd.DataFrame(index=bench_position_weight.index,columns=bench_position_weight.columns)
        daily_weight = None
        

        for i in range(1,len(bench_position_weight)):


            date = bench_position_weight.index[i]
            last_date = self.bench_exposure.loc[:date].index[-1]
            last_freq_date = bench_position_weight.index[i-1]
            try:
                next_date = bench_position_weight.index[i+1]
            except:
                next_date = bench_position_weight.index[-1]
            hold_df = self.ret_D_df.loc[date:next_date].iloc[:-1]



            B,R2_dict,resudial,ret_W_df_date = self.exposure_cal(last_date,self.ret_W_df_1,self.HF_macro_factor_2)
            B = B.values
            weight_bench = np.array(bench_position_weight.loc[date])
            n = len(weight_bench)

            last_bench_exposure = np.array(self.bench_exposure.loc[last_date].values)
            change_exposure =np.array(np.array(HF_macro_predict_1.loc[:last_freq_date].iloc[-1]) * self.sigma_beta)

            target_exposure = last_bench_exposure + change_exposure
            lambda_ = 0.1
            sigma = ret_W_df_date.cov().values
            w = cp.Variable(n)
            w0 = weight_bench
            T = target_exposure
            Q = resudial.cov().values

            delta_w = w-w0

            #构造目标函数各部分
            term1 = cp.quad_form(delta_w,sigma)
            term2 = (1-lambda_)*cp.norm(B.T@w-T,2)
            term3 = lambda_*cp.quad_form(w,Q)
            obj = cp.Minimize(term1+term2+term3)

            #约束
            contraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= 1,
                delta_w >= -1,
                delta_w <= 1
            ]

            #定义问题
            problem = cp.Problem(obj,contraints)
            result = problem.solve()
            dw = (w.value - w0)
            delta_w_series.loc[date] = dw
            w_series.loc[date] = w.value

            daily_w = self.calculate_dynamic_weight(hold_df, np.matrix(w.value).T, date)
            if daily_weight is None:
                daily_weight = daily_w
            else:
                daily_weight = pd.concat([daily_weight,daily_w])

            self.delta_w_series = delta_w_series
            self.w_series = w_series
            self.daily_weight = daily_weight
            self.logger.info(f"回测日期:{date} 权重:{w.value}")

        self.daily_weight.to_excel('./excel/MFH_optimal_weight_series.xlsx')
        self.logger.info("计算完成")
        self.logger.info("="*60)

    
    def get_results_summary(self,plot=True):
        """
        获取回测结果摘要
        
        Returns:
        --------
        summary : pd.DataFrame
            结果摘要表格
        """
        self.logger.info("="*60)
        self.logger.info("获取MFH最优权重回测结果摘要")
        self.logger.info("="*60)
        bench_weight,new_weight = self.index_common(self.bench_weight_df,self.daily_weight,print_info=False)
        ret_df =self.price_df_d[self.config.BENCHMARK_ASSET][self.price_df_d.index.isin(bench_weight.index)].pct_change()
        ret_df.fillna(0,inplace=True)
        ret_bench = (ret_df * bench_weight).sum(axis=1)
        ret_new = (ret_df*new_weight).sum(axis=1)
        ret_df_list = {"基准":ret_bench,"MFH":ret_new}
        summary_data = []
        pv_list = {}
        for strategy_name, ret_df in ret_df_list.items():
            pv = (1+ret_df).cumprod()
            returns = ret_df
            pv_list[strategy_name] = pv
            
            summary_data.append({
                '策略名称': strategy_name,
                '最终净值': pv.iloc[-1],
                '累计收益率': pv.iloc[-1] / pv.iloc[0] - 1,
                '年化收益率': (pv.iloc[-1] / pv.iloc[0]) ** (252 / len(pv)) - 1,
                '年化波动率': returns.std() * np.sqrt(252),
                '最大回撤': self._calculate_max_drawdown(returns),
                '夏普比率': (((pv.iloc[-1] / pv.iloc[0]) ** (252 / len(pv)) - 1) / (returns.std() * np.sqrt(252)))
            })
        self.logger.info(f'模型总体评价:\n{pd.DataFrame(summary_data).T}')
        if plot:
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            # 1. 净值曲线对比
            fig, axes = plt.subplots(2, 1, figsize=(15, 10))
            
            # 净值曲线
            for strategy_name, pv in pv_list.items():
                axes[0].plot(pv.index, pv.values, label=strategy_name, linewidth=2)
            axes[0].set_title('策略净值曲线对比', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('日期')
            axes[0].set_ylabel('净值')
            axes[0].legend(loc='upper left')
            axes[0].grid(True, alpha=0.3)
            
            # RP策略权重变化


            axes[1].stackplot(
                self.daily_weight.index,
                self.daily_weight.T,
                labels=self.daily_weight.columns,
                alpha=0.8
            )
            axes[1].set_title('MFH最优权重变化', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('日期')
            axes[1].set_ylabel('权重')
            axes[1].set_yticks(np.arange(0, 1.1, 0.2))
            axes[1].legend(loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
            plt.tight_layout()
            plt.show()
            self.logger.info("="*60)
        self.pv_dict = pv_list
        self.returns_dict = ret_df_list
        return pd.DataFrame(summary_data)
    
    def _calculate_max_drawdown(self, returns):
        """
        计算最大回撤
        
        Parameters:
        -----------
        returns : pd.Series
            收益率序列
            
        Returns:
        --------
        max_dd : float
            最大回撤
        """
        bench_weight,new_weight = self.index_common(self.bench_weight_df,self.daily_weight,print_info=False)
        ret_df =self.price_df_d[self.config.BENCHMARK_ASSET][self.price_df_d.index.isin(bench_weight.index)].pct_change()

        cumulative_wealth = (1 + returns).cumprod()
        running_max = cumulative_wealth.expanding().max()
        drawdown = (cumulative_wealth - running_max) / running_max
        max_drawdown = drawdown.min()
        return max_drawdown
    
    def evalute_results(self,save_results=True):
        """评价回测结果"""
        self.logger.info( "="*60)
        self.logger.info("开始评价回测结果")
        self.logger.info("="*60)
        

        #整体评价
        self.total_summary = self.get_results_summary()
        pv_dict = self.pv_dict
    
        # RP策略的年度分析
        strategy_returns = self.returns_dict['MFH']
        annual_df = PeriodAnalyzer.annual_analysis(
            strategy_returns,
            var_windows=self.config.VAR_YEAR_WINDOWS,
            max_loss_limit=self.config.MAX_LOSS_LIMIT
        )

        self.annual_df = annual_df
            
        # 打印年度报告
        ReportGenerator.print_annual_report(annual_df, 'RP策略年度分析')
        self.logger.info(f'模型分年度评价:\n{annual_df}')

        

    def run_workflow(self):
        self.logger.info("="*60)
        self.logger.info("MFH工作流开始")
        self.logger.info("="*60)
        self.exposure_cal_parallel()
        self.Macro_stat_predict()
        self.MFH_optimal()

        self.evalute_results()
        self.logger.info(f'回测结束')


 


if __name__ == '__main__':
    config = MFHConfig()
    mfh = MFH(config)
    mfh.run_workflow()
    import pickle
    with open('MFH_total_summary.pkl', 'wb') as f:
        pickle.dump(mfh.total_summary, f)




