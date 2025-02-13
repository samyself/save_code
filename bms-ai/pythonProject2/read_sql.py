import pandas as pd
from pyhive import hive

def select_pyhive(sql, hostname, username):
    conn = hive.Connection(host=hostname, port=80, username=username)
    cur = conn.cursor()
    try:
        # cursor.execute(sql, async_=True) # python3.7以下版本使用cursor.execute(sql, async=True)
        # print(cursor.fetchall())
        df = pd.read_sql(sql, conn)
        # df.dropna(inplace=True)
        return df
    finally:
        if conn:
            conn.close()

# 量产环境和研发环境的配置
host_pro = 'proxy-service-thrift-nc4cloudprc-dp.api.xiaomi.net'
user_pro = '61af1e94d55c4902b2354902e9535e9b'
host_pre = 'proxy-service-thrift-nc4prc-dp.api.xiaomi.net'
user_pre = 'dcf72eb9e82c4abbb1618781b11798e5'

mode = 'pro'
if mode == 'pre':
    hostname = host_pre
    username = user_pre
else:
    hostname = host_pro
    username = user_pro

sql = f"""
SELECT
    *
FROM iceberg_nc4cloudprc_hadoop.bms.lfp_ai_soc_result
limit 5000
"""
result_df = select_pyhive(sql, hostname, username)

csv_path = '../data/csv_res/Soc_lstm_res.csv'
result_df.to_csv(csv_path, index=False)