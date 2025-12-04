'''
配置文件 - 管理MFH所使用的参数
'''

class MFHConfig:
    END_DATE =  '2024-03-15'
    CHG_FREQ = 'MS' #月频换仓，月初调仓
    BENCHMARK_ASSET = ['000300_SH',
                       '000905_SH',
                       'CBA00601_CS',
                       'CBA02001_CS',
                       '000832_CSI',
                       'NH0200_NHF',
                       'NH0300_NHF',
                       'B_IPE',
                       'AU9999_SGE',
                       'USDCH_FX'
                       ]
    
    BENCHMARK_MODEL_WEIGHT_PATH = './RP/RP_weight_df_20251203.xlsx'
    BENCHMARK_MODEL_POSITION_PATH = './RP/position_df.xlsx'

    # ============ 日志配置 ============
    # 日志文件路径
    LOG_FILE_PATH = './log/'
    
    # 日志文件名前缀
    LOG_FILE_PREFIX = 'MFH_model'
    
    # 日志级别 (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    LOG_LEVEL = 'INFO'
    # 是否保存评价结果
    SAVE_EVAL_RESULTS = True
    VAR_YEAR_WINDOWS = 200_000_000
    MAX_LOSS_LIMIT = 5