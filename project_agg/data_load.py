import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import datetime
from WindPy import w
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api  as sm 
from statsmodels.stats.outliers_influence  import variance_inflation_factor
from LF_macro_factor_cal import LF_macro_factor_cal

from Util_Fin import easy_manager
from Util_Fin import logger_util

from config import Config

def raw_macro_data_to_db():
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")
    code_str = ','.join(Config.RAW_MACRO_DATA_LIST.values())
    # 从Wind读取数据 - 批量
    macro_data = w.edb(code_str,
                    Config.RAW_MACRO_DATA_START_DATE, 
                    Config.RAW_MACRO_DATA_END_DATE,
                    "Fill=Previous",
                    "Period=M",
                    usedf=True)[1]
    macro_data.index=pd.to_datetime(macro_data.index)
    macro_data_M = macro_data.resample('ME').last().dropna(how='all')
    # 过滤掉晚于今天的数据
    macro_data_M = _filter_future_dates(macro_data_M, main_logger)
    with easy_manager.EasyManager(database='macro_data_base') as em:
        em.create_table(table_name='raw_macro_data_m',dataframe=macro_data_M,overwrite=True)

def raw_macro_data_to_db_update():
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")

    with easy_manager.EasyManager(database='macro_data_base') as em:
        df = em.load_table(table_name='raw_macro_data_m',limit=-10)
        update_start_date = df['index'].iloc[-1] - datetime.timedelta(days=31)
        last_code_list = df.columns.tolist()[1:]
        today = datetime.date.today()
        current_month = datetime.date(today.year, today.month, 1) - datetime.timedelta(days=1)
        current_month = current_month.strftime('%Y-%m-%d')
        code_list = Config.RAW_MACRO_DATA_LIST.values()
        # code_str = ','.join(Config.RAW_MACRO_DATA_LIST.values())
        last_code_str = ','.join(last_code_list)
        main_logger.info("正在检测本次更新是否存在新增指标...")
        code_add_list = list(set(code_list) - set(last_code_list))
        len_code_add_list = len(code_add_list)
        code_add_str = ','.join(code_add_list)

        if len_code_add_list > 0:
            main_logger.info(f"本次更新存在新增指标,新增指标为: {code_add_list}")
            new_code_list = last_code_list + code_add_list
            # code_str = ','.join(new_code_list)
            main_logger.info(f"开始录入存量指标数据...")
            macro_data = w.edb(last_code_str,
                            update_start_date, 
                            current_month,
                            "Fill=Previous",
                            "Period=M",
                            usedf=True)[1]
            macro_data.index=pd.to_datetime(macro_data.index)
            macro_data_M = macro_data.resample('ME').last().dropna(how='all')
            macro_data_M = macro_data_M[last_code_list]
            # 过滤掉晚于今天的数据
            macro_data_M = _filter_future_dates(macro_data_M, main_logger)
            print(macro_data_M)
            if len(macro_data_M) > 0:
                em.insert_data(table_name='raw_macro_data_m',dataframe=macro_data_M,mode='update')
            else:
                main_logger.info("本次更新不存在无需录入存量指标数据.")
            main_logger.info(f"开始录入新增指标数据...")
            macro_data_add = w.edb(code_add_str,
                            Config.RAW_MACRO_DATA_START_DATE, 
                            current_month,
                            "Fill=Previous",
                            "Period=M",
                            usedf=True)[1]
            macro_data_add.columns = code_add_list
            macro_data_add.index=pd.to_datetime(macro_data_add.index)
            macro_data_M_add = macro_data_add.resample('ME').last().dropna(how='all')
            macro_data_M_add = macro_data_M_add[code_add_list]
            # 过滤掉晚于今天的数据
            macro_data_M_add = _filter_future_dates(macro_data_M_add, main_logger)
            em.add_columns(table_name='raw_macro_data_m',dataframe=macro_data_M_add,merge_on_index=True)
            main_logger.info(f"新增指标数据录入完成.")
            main_logger.info(f"本次更新完成.")

        else:
            main_logger.info("本次更新不存在新增指标.")
            main_logger.info(f"开始录入存量指标数据...")
            macro_data = w.edb(last_code_str,
                            update_start_date, 
                            current_month,
                            "Fill=Previous",
                            "Period=M",
                            usedf=True)[1]
            macro_data.index=pd.to_datetime(macro_data.index)
            macro_data_M = macro_data.resample('ME').last().dropna(how='all')
            macro_data_M = macro_data_M[last_code_list]
            # 过滤掉晚于今天的数据
            macro_data_M = _filter_future_dates(macro_data_M, main_logger)
            print(macro_data_M)
            if len(macro_data_M) > 0:
                em.insert_data(table_name='raw_macro_data_m',dataframe=macro_data_M,mode='update')
            else:
                main_logger.info("本次更新不存在无需录入存量指标数据.")
            main_logger.info(f"本次更新完成.")


def LF_macro_data_to_db_update():
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    main_logger.info("开始计算低频宏观因子...")
    with easy_manager.EasyManager(database='macro_data_base') as em:
        df = em.load_table(table_name='raw_macro_data_m',order_by='index',ascending=True)
        df.set_index("index",inplace=True)
        df.index = pd.to_datetime(df.index)
        print("原始指标如下:")
        print(df.tail(10))

        lf_macro_factor_cal = LF_macro_factor_cal(factor_plot=True,dataframe=df)
        LF_data = lf_macro_factor_cal.cal_all_factors()
        # em.create_table(table_name='lf_macro_data',dataframe=LF_data,overwrite=True)
        LF_data_new = LF_data.tail(2)
        em.insert_data(table_name='lf_macro_data',dataframe=LF_data_new,mode='skip')
        main_logger.info(f"本次低频指标计算完成,本次更新完成.")

def _normalize_code(code):
    """
    将Wind代码转换为数据库列名格式（特殊符号转为下划线）
    """
    return code.replace('.', '_').replace('-', '_')

def _filter_future_dates(dataframe, logger=None):
    """
    过滤掉晚于今天的数据
    
    参数:
    dataframe: pd.DataFrame, 索引为日期的数据框
    logger: logger对象，用于记录日志
    
    返回:
    pd.DataFrame: 过滤后的数据框
    """
    today = pd.Timestamp(datetime.date.today())
    
    # 检查是否有晚于今天的数据
    future_dates = dataframe.index > today
    if future_dates.any():
        future_count = future_dates.sum()
        if logger:
            logger.warning(f"发现 {future_count} 条晚于今天的数据，将被过滤掉")
            logger.warning(f"最晚日期: {dataframe.index[future_dates].max()}")
        
        # 只保留今天及之前的数据
        dataframe = dataframe[dataframe.index <= today]
        
        if logger:
            logger.info(f"过滤后数据范围: {dataframe.index.min()} 到 {dataframe.index.max()}")
    
    return dataframe

def hf_asset_data_to_db():
    """
    初始化高频资产数据表，从Wind获取周频资产价格数据
    """
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")
    main_logger.info("开始批量获取高频资产数据...")
    
    # 获取wsd数据（股票/指数价格）
    wsd_code_str = ','.join(Config.HF_ASSET_WSD_LIST.keys())
    main_logger.info(f"正在获取WSD数据: {wsd_code_str}")
    wsd_data = w.wsd(wsd_code_str, 
                     'close',
                     Config.HF_ASSET_DATA_START_DATE, 
                     Config.HF_ASSET_DATA_END_DATE,
                     f"Period={Config.HF_ASSET_DATA_PERIOD}",
                     usedf=True)[1]
    wsd_data.index = pd.to_datetime(wsd_data.index)
    # 重采样为周频，取最后一个值
    wsd_data_w = wsd_data.resample('W').last()
    main_logger.info(f"WSD数据获取完成，形状: {wsd_data_w.shape}")
    
    # 获取edb数据（经济数据）
    edb_code_str = ','.join(Config.HF_ASSET_EDB_LIST.keys())
    main_logger.info(f"正在获取EDB数据: {edb_code_str}")
    edb_data = w.edb(edb_code_str,
                     Config.HF_ASSET_DATA_START_DATE, 
                     Config.HF_ASSET_DATA_END_DATE,
                     "Fill=Previous",
                     f"Period={Config.HF_ASSET_DATA_PERIOD}",
                     usedf=True)[1]
    edb_data.index = pd.to_datetime(edb_data.index)
    # 重采样为周频，取最后一个值
    edb_data_w = edb_data.resample('W').last()
    main_logger.info(f"EDB数据获取完成，形状: {edb_data_w.shape}")
    
    # 合并数据
    asset_data_w = pd.concat([wsd_data_w, edb_data_w], axis=1)
    asset_data_w = asset_data_w.dropna(how='all')
    main_logger.info(f"合并后数据形状: {asset_data_w.shape}")
    print("高频资产数据预览:")
    print(asset_data_w.head())
    print(asset_data_w.tail())
    
    # 过滤掉晚于今天的数据
    asset_data_w = _filter_future_dates(asset_data_w, main_logger)
    
    # 存储到数据库
    with easy_manager.EasyManager(database='macro_data_base') as em:
        em.create_table(table_name='hf_asset_data_w', dataframe=asset_data_w, overwrite=True)
        main_logger.info("高频资产数据已存储到数据库")

def hf_asset_data_to_db_update():
    """
    更新高频资产数据表
    """
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")
    main_logger.info("开始更新高频资产数据...")
    
    with easy_manager.EasyManager(database='macro_data_base') as em:
        # 获取现有数据的最后日期
        df = em.load_table(table_name='hf_asset_data_w', limit=-10)
        update_start_date = df['index'].iloc[-1] - datetime.timedelta(days=14)  # 往前推2周
        last_code_list_db = df.columns.tolist()[1:]  # 排除index列，这是数据库中的列名（已转换为下划线）
        
        # 计算更新结束日期（当前日期所在周的周日）
        today = datetime.date.today()
        days_since_sunday = (today.weekday() + 1) % 7
        current_week_end = today - datetime.timedelta(days=days_since_sunday)
        current_week_end_str = current_week_end.strftime('%Y-%m-%d')
        
        main_logger.info(f"更新起始日期: {update_start_date}")
        main_logger.info(f"更新结束日期: {current_week_end_str}")
        
        # 获取原始代码列表（带特殊符号）
        wsd_code_list_raw = list(Config.HF_ASSET_WSD_LIST.keys())
        edb_code_list_raw = list(Config.HF_ASSET_EDB_LIST.keys())
        all_code_list_raw = wsd_code_list_raw + edb_code_list_raw
        
        # 将原始代码转换为数据库格式（用于比较）
        wsd_code_list_norm = [_normalize_code(c) for c in wsd_code_list_raw]
        edb_code_list_norm = [_normalize_code(c) for c in edb_code_list_raw]
        all_code_list_norm = wsd_code_list_norm + edb_code_list_norm
        
        # 创建原始代码到规范化代码的映射
        raw_to_norm = {raw: norm for raw, norm in zip(all_code_list_raw, all_code_list_norm)}
        norm_to_raw = {norm: raw for raw, norm in zip(all_code_list_raw, all_code_list_norm)}
        
        # 比较时使用规范化的名称
        code_add_list_norm = list(set(all_code_list_norm) - set(last_code_list_db))
        code_add_list_raw = [norm_to_raw[code] for code in code_add_list_norm]  # 转回原始代码用于调用Wind API
        
        main_logger.info(f"数据库现有列: {last_code_list_db}")
        main_logger.info(f"配置文件代码(规范化): {all_code_list_norm}")
        main_logger.info(f"新增代码(规范化): {code_add_list_norm}")
        main_logger.info(f"新增代码(原始): {code_add_list_raw}")
        
        len_code_add_list = len(code_add_list_raw)
        
        if len_code_add_list > 0:
            main_logger.info(f"本次更新存在新增指标: {code_add_list_raw}")
            # 分离WSD和EDB代码（使用原始代码）
            code_add_wsd = [c for c in code_add_list_raw if c in wsd_code_list_raw]
            code_add_edb = [c for c in code_add_list_raw if c in edb_code_list_raw]
            
            # 先更新存量指标
            main_logger.info("开始更新存量指标数据...")
            # 分离现有的WSD和EDB代码（需要从规范化名称转回原始名称）
            last_wsd_raw = [norm_to_raw.get(c, c) for c in last_code_list_db if norm_to_raw.get(c, c) in wsd_code_list_raw]
            last_edb_raw = [norm_to_raw.get(c, c) for c in last_code_list_db if norm_to_raw.get(c, c) in edb_code_list_raw]
            # 更新WSD数据（使用原始代码调用API）
            if last_wsd_raw:
                wsd_code_str = ','.join(last_wsd_raw)
                wsd_data = w.wsd(wsd_code_str,
                                'close',
                                update_start_date, 
                                current_week_end_str,
                                f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                usedf=True)[1]
                wsd_data.index = pd.to_datetime(wsd_data.index)
                wsd_data_w = wsd_data.resample('W').last()
            else:
                wsd_data_w = pd.DataFrame()
            
            # 更新EDB数据（使用原始代码调用API）
            if last_edb_raw:
                edb_code_str = ','.join(last_edb_raw)
                edb_data = w.edb(edb_code_str,
                                update_start_date, 
                                current_week_end_str,
                                "Fill=Previous",
                                f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                usedf=True)[1]
                edb_data.index = pd.to_datetime(edb_data.index)
                edb_data_w = edb_data.resample('W').last()
            else:
                edb_data_w = pd.DataFrame()
            
            # 合并并更新
            if not wsd_data_w.empty or not edb_data_w.empty:
                asset_data_w = pd.concat([wsd_data_w, edb_data_w], axis=1)
                # 数据库列名已经是规范化格式，需要选择对应的列
                asset_data_w = asset_data_w[last_wsd_raw + last_edb_raw]
                asset_data_w = asset_data_w.dropna(how='all')
                # 过滤掉晚于今天的数据
                asset_data_w = _filter_future_dates(asset_data_w, main_logger)
                print("存量指标更新数据:")
                print(asset_data_w)
                if len(asset_data_w) > 0:
                    em.insert_data(table_name='hf_asset_data_w', dataframe=asset_data_w, mode='update')
                    main_logger.info("存量指标数据更新完成")
                else:
                    main_logger.info("本次无需更新存量指标数据")
            
            # 处理新增指标
            main_logger.info("开始获取新增指标数据...")
            # 获取新增WSD数据
            if code_add_wsd:
                wsd_add_str = ','.join(code_add_wsd)
                wsd_data_add = w.wsd(wsd_add_str,
                                    'close',
                                    Config.HF_ASSET_DATA_START_DATE, 
                                    current_week_end_str,
                                    f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                    usedf=True)[1]
                wsd_data_add.columns = code_add_wsd
                wsd_data_add.index = pd.to_datetime(wsd_data_add.index)
                wsd_data_add_w = wsd_data_add.resample('W').last()
            else:
                wsd_data_add_w = pd.DataFrame()
            
            # 获取新增EDB数据
            if code_add_edb:
                edb_add_str = ','.join(code_add_edb)
                edb_data_add = w.edb(edb_add_str,
                                    Config.HF_ASSET_DATA_START_DATE, 
                                    current_week_end_str,
                                    "Fill=Previous",
                                    f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                    usedf=True)[1]
                edb_data_add.columns = code_add_edb
                edb_data_add.index = pd.to_datetime(edb_data_add.index)
                edb_data_add_w = edb_data_add.resample('W').last()
            else:
                edb_data_add_w = pd.DataFrame()
            
            # 合并新增数据
            if not wsd_data_add_w.empty or not edb_data_add_w.empty:
                asset_data_add_w = pd.concat([wsd_data_add_w, edb_data_add_w], axis=1)
                asset_data_add_w = asset_data_add_w.dropna(how='all')
                # 过滤掉晚于今天的数据
                asset_data_add_w = _filter_future_dates(asset_data_add_w, main_logger)
                em.add_columns(table_name='hf_asset_data_w', dataframe=asset_data_add_w, merge_on_index=True)
                main_logger.info("新增指标数据录入完成")
            
            main_logger.info("本次更新完成")
        else:
            main_logger.info("本次更新不存在新增指标")
            main_logger.info("开始更新存量指标数据...")
            
            # 分离WSD和EDB代码（从规范化名称转回原始名称）
            wsd_codes_raw = [norm_to_raw.get(c, c) for c in last_code_list_db if norm_to_raw.get(c, c) in wsd_code_list_raw]
            edb_codes_raw = [norm_to_raw.get(c, c) for c in last_code_list_db if norm_to_raw.get(c, c) in edb_code_list_raw]
            
            # 更新WSD数据（使用原始代码调用API）
            if wsd_codes_raw:
                wsd_code_str = ','.join(wsd_codes_raw)
                wsd_data = w.wsd(wsd_code_str,
                                'close',
                                update_start_date, 
                                current_week_end_str,
                                f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                usedf=True)[1]
                wsd_data.index = pd.to_datetime(wsd_data.index)
                wsd_data_w = wsd_data.resample('W').last()
            else:
                wsd_data_w = pd.DataFrame()
            
            # 更新EDB数据（使用原始代码调用API）
            if edb_codes_raw:
                edb_code_str = ','.join(edb_codes_raw)
                edb_data = w.edb(edb_code_str,
                                update_start_date, 
                                current_week_end_str,
                                "Fill=Previous",
                                f"Period={Config.HF_ASSET_DATA_PERIOD}",
                                usedf=True)[1]
                edb_data.index = pd.to_datetime(edb_data.index)
                edb_data_w = edb_data.resample('W').last()
            else:
                edb_data_w = pd.DataFrame()
            
            # 合并并更新
            if not wsd_data_w.empty or not edb_data_w.empty:
                asset_data_w = pd.concat([wsd_data_w, edb_data_w], axis=1)
                # 选择对应的列（原始代码格式）
                asset_data_w = asset_data_w[wsd_codes_raw + edb_codes_raw]
                asset_data_w = asset_data_w.dropna(how='all')
                # 过滤掉晚于今天的数据
                asset_data_w = _filter_future_dates(asset_data_w, main_logger)
                print("更新数据:")
                print(asset_data_w)
                if len(asset_data_w) > 0:
                    em.insert_data(table_name='hf_asset_data_w', dataframe=asset_data_w, mode='update')
                    main_logger.info("数据更新完成")
                else:
                    main_logger.info("本次无需更新数据")
            
            main_logger.info("本次更新完成")

if __name__ == '__main__':
    # 初始化数据（首次运行时使用）
    raw_macro_data_to_db()
    # hf_asset_data_to_db()
    
    # 日常更新数据
    raw_macro_data_to_db_update()
    # hf_asset_data_to_db_update()
    # LF_macro_data_to_db_update()
    
    # 查看数据
    with easy_manager.EasyManager(database='macro_data_base') as em:
        # em.drop_table(table_name='hf_asset_data_w')
        # 原始宏观数据
        # df = em.load_table(table_name='raw_macro_data_m',order_by='index',ascending=True)
        # df.set_index("index",inplace=True)
        # df.index = pd.to_datetime(df.index)
        # print("="*60)
        # print("原始宏观指标数据:")
        # print(df.tail(10))
        
        # # 低频宏观因子
        # df = em.load_table(table_name='lf_macro_data',order_by='index',ascending=True)
        # print("\n" + "="*60)
        # print("低频宏观因子:")
        # print(df.tail(10))
        
        # 高频资产数据
        df = em.load_table(table_name='hf_asset_data_w',order_by='index',ascending=True)
        df.set_index("index",inplace=True)
        df.index = pd.to_datetime(df.index)
        print("\n" + "="*60)
        print("高频资产数据:")
        print(df.tail(10))
