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

DAILY_ASSET_TABLE_NAME = 'daily_asset_price'

def raw_macro_data_to_db():
    '''
    从Wind读取原始宏观数据并存储到数据库中
    -------------------------------
    数据类型: 月频的宏观数据
    数据作用: 用于计算低频宏观因子
    '''
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
    '''
    从Wind更新原始宏观数据并存储到数据库中
    -------------------------------
    数据类型: 月频的宏观数据
    数据作用: 用于计算低频宏观因子
    -------------------------------
    注意: 需要在第一次先运行一次raw_macro_data_to_db()，之后只运行raw_macro_data_to_db_update()即可
    '''
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
    '''
    从Wind数据库中读取原始宏观数据并计算低频宏观因子，最后将结果存储到本地数据库中
    -------------------------------
    存储数据类型: 月频的宏观因子
    '''
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

def daily_asset_price_to_db():
    """
    初始化日度资产价格数据表，使用Part 2资产列表从Wind获取日度收盘价数据
    """
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")
    main_logger.info("开始获取日度资产价格数据...")

    start_date = Config.HF_ASSET_DATA_START_DATE
    end_date = Config.HF_ASSET_DATA_END_DATE

    wsd_data = pd.DataFrame()
    if Config.ASSET_LIST_WSD:
        wsd_code_str = ','.join(Config.ASSET_LIST_WSD)
        main_logger.info(f"正在获取WSD数据: {wsd_code_str}")
        wsd_data = w.wsd(wsd_code_str,
                         'close',
                         start_date,
                         end_date,
                         usedf=True)[1]
        wsd_data.index = pd.to_datetime(wsd_data.index)
        wsd_data = wsd_data[Config.ASSET_LIST_WSD]
    else:
        main_logger.info("未配置WSD日度资产代码")

    edb_data = pd.DataFrame()
    if Config.ASSET_LIST_EDB:
        edb_code_str = ','.join(Config.ASSET_LIST_EDB.keys())
        main_logger.info(f"正在获取EDB数据: {edb_code_str}")
        edb_data = w.edb(edb_code_str,
                         start_date,
                         end_date,
                         "Fill=Previous",
                         usedf=True)[1]
        edb_data.index = pd.to_datetime(edb_data.index)
        edb_data = edb_data[list(Config.ASSET_LIST_EDB.keys())]
        edb_data.rename(columns=Config.ASSET_LIST_EDB, inplace=True)
    else:
        main_logger.info("未配置EDB日度资产代码")

    asset_data_d = pd.concat([wsd_data, edb_data], axis=1)
    asset_data_d = asset_data_d.dropna(how='all')
    asset_data_d = _filter_future_dates(asset_data_d, main_logger)
    main_logger.info(f"日度资产数据获取完成，形状: {asset_data_d.shape}")

    with easy_manager.EasyManager(database='macro_data_base') as em:
        em.create_table(table_name=DAILY_ASSET_TABLE_NAME,
                        dataframe=asset_data_d,
                        overwrite=True)
        main_logger.info("日度资产数据已写入数据库")

def daily_asset_price_to_db_update():
    """
    更新日度资产价格数据表，支持新增资产代码和存量代码更新
    """
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")

    with easy_manager.EasyManager(database='macro_data_base') as em:
        df = em.load_table(table_name=DAILY_ASSET_TABLE_NAME, limit=-10)
        if df is None or len(df) == 0:
            main_logger.error("数据库中不存在日度资产数据，请先运行daily_asset_price_to_db()初始化数据")
            return
        last_date = pd.to_datetime(df['index'].iloc[-1])
        update_start_date = (last_date - datetime.timedelta(days=7)).strftime('%Y-%m-%d')
        last_code_list_db = df.columns.tolist()[1:]

    now = datetime.datetime.now()
    if now.hour >= 15 and now.minute >= 1:
        today = datetime.date.today().strftime('%Y-%m-%d')
    else:
        today = (datetime.date.today() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')
    main_logger.info(f"更新起始日期: {update_start_date}，结束日期: {today}")

    wsd_code_raw = list(Config.ASSET_LIST_WSD)
    wsd_code_norm = [_normalize_code(code) for code in wsd_code_raw]
    raw_to_norm = dict(zip(wsd_code_raw, wsd_code_norm))
    norm_to_raw = dict(zip(wsd_code_norm, wsd_code_raw))

    edb_code_map = dict(Config.ASSET_LIST_EDB)
    edb_alias_list = list(edb_code_map.values())
    edb_alias_to_raw = {alias: raw for raw, alias in edb_code_map.items()}

    existing_wsd_norm = [code for code in last_code_list_db if code in norm_to_raw]
    existing_wsd_raw = [norm_to_raw[code] for code in existing_wsd_norm]
    existing_edb_alias = [code for code in last_code_list_db if code in edb_alias_to_raw]
    existing_edb_raw = [edb_alias_to_raw[alias] for alias in existing_edb_alias]

    add_wsd_norm = list(set(wsd_code_norm) - set(last_code_list_db))
    add_wsd_raw = [norm_to_raw[code] for code in add_wsd_norm]
    add_edb_alias = list(set(edb_alias_list) - set(last_code_list_db))
    add_edb_raw = [edb_alias_to_raw[alias] for alias in add_edb_alias]

    main_logger.info(f"数据库现有字段: {last_code_list_db}")
    main_logger.info(f"WSD新增代码: {add_wsd_raw}")
    main_logger.info(f"EDB新增代码: {add_edb_raw}")

    def _fetch_wsd_data(code_list, start_dt, end_dt):
        if not code_list:
            return pd.DataFrame()
        data = w.wsd(','.join(code_list),
                     'close',
                     start_dt,
                     end_dt,
                     usedf=True)[1]
        data.index = pd.to_datetime(data.index)
        data = pd.DataFrame(data)
        data.columns = code_list
        return data[code_list]

    def _fetch_edb_data(code_list, aliases, start_dt, end_dt):
        if not code_list:
            return pd.DataFrame()
        data = w.edb(','.join(code_list),
                     start_dt,
                     end_dt,
                     "Fill=Previous",
                     usedf=True)[1]
        data.index = pd.to_datetime(data.index)
        rename_map = {raw: alias for raw, alias in zip(code_list, aliases)}
        data = data[code_list]
        data.rename(columns=rename_map, inplace=True)
        return data

    with easy_manager.EasyManager(database='macro_data_base') as em:
        # 更新存量代码
        wsd_existing_df = _fetch_wsd_data(existing_wsd_raw, update_start_date, today)
        edb_existing_df = _fetch_edb_data(existing_edb_raw, existing_edb_alias, update_start_date, today)
        existing_asset_df = pd.concat([wsd_existing_df, edb_existing_df], axis=1)
        existing_asset_df = existing_asset_df.dropna(how='all')
        existing_asset_df = _filter_future_dates(existing_asset_df, main_logger)
        if not existing_asset_df.empty:
            main_logger.info("开始更新存量资产代码数据")
            em.insert_data(table_name=DAILY_ASSET_TABLE_NAME,
                           dataframe=existing_asset_df,
                           mode='update')
            main_logger.info("存量资产代码更新完成")
        else:
            main_logger.info("本次无存量资产数据需要更新")

        # 处理新增代码
        if add_wsd_raw or add_edb_raw:
            main_logger.info("开始处理新增资产代码")
            wsd_new_df = _fetch_wsd_data(add_wsd_raw,
                                         Config.HF_ASSET_DATA_START_DATE,
                                         today)
            edb_new_alias = [Config.ASSET_LIST_EDB[raw] for raw in add_edb_raw]
            edb_new_df = _fetch_edb_data(add_edb_raw,
                                         edb_new_alias,
                                         Config.HF_ASSET_DATA_START_DATE,
                                         today)
            new_asset_df = pd.concat([wsd_new_df, edb_new_df], axis=1)
            new_asset_df = new_asset_df.dropna(how='all')
            new_asset_df = _filter_future_dates(new_asset_df, main_logger)
            if not new_asset_df.empty:
                em.add_columns(table_name=DAILY_ASSET_TABLE_NAME,
                               dataframe=new_asset_df,
                               merge_on_index=True)
                main_logger.info("新增资产代码写入完成")
            else:
                main_logger.info("新增资产代码暂无有效数据")
        else:
            main_logger.info("本次更新无新增资产代码")

def hf_asset_data_to_db():
    """
    初始化周频资产数据表，从Wind获取周频资产价格数据并存储到本地数据库中
    -------------------------------
    数据类型: 周频的资产价格数据
    数据作用: 用于计算周频宏观因子
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
        main_logger.info("高频宏观因子制作-资产数据已存储到数据库")

def hf_asset_data_to_db_update():
    """
    从Wind更新周频资产价格数据并存储到数据库中
    -------------------------------
    数据类型: 周频的资产价格数据
    数据作用: 用于计算周频宏观因子
    -------------------------------
    注意: 需要在第一次先运行一次hf_asset_data_to_db()，之后只运行hf_asset_data_to_db_update()即可
    """
    main_logger = logger_util.setup_logger("data_load_from_wind",'./')
    w.start()
    main_logger.info("WindPy started")
    main_logger.info("开始更新高频宏观因子制作所需资产数据...")
    
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
    # raw_macro_data_to_db()
    # hf_asset_data_to_db()
    # daily_asset_price_to_db()
    
    # 日常更新数据
    # raw_macro_data_to_db_update()
    # hf_asset_data_to_db_update()
    daily_asset_price_to_db_update()

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
        df = em.load_table(table_name='daily_asset_price_d',order_by='index',ascending=True)
        df.set_index("index",inplace=True)
        df.index = pd.to_datetime(df.index)
        print("\n" + "="*60)
        print("日频资产价格数据:")
        print(df.tail(10))
