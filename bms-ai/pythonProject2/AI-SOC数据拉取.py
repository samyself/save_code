### AI-SOC模型数据拉取
import pandas as pd
from pyhive import hive

def select_pyhive(sql, hostname, username):
    conn = hive.Connection(host=hostname, port=80, username=username)
    cur = conn.cursor()
    try:
        df = pd.read_sql(sql, conn)
        # df.dropna(inplace=True)
        return df
    finally:
        if conn:
            conn.close()

def maincode(sql, date1, date2, vin, savepath):
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

    sql_type = {
    # 提取AI-SOC结果表数据
    'sql1' : f"""
            SELECT
                *
            FROM iceberg_nc4cloudprc_hadoop.bms.lfp_ai_soc_result
            where date >= '{date1}' and DATE <= '{date2}' 
            """,
    # 提取AI-SOC结果表+hvbattcellmaxsocerr字段数据
    'sql2' : f"""
            WITH A AS (
                SELECT
                *
                FROM iceberg_nc4cloudprc_hadoop.bms.lfp_ai_soc_result
                WHERE DATE >= '{date1}' and DATE <= '{date2}' ),
            B AS (
                SELECT
                    vid,
                    ts,
                    hvbattcellmaxsocerr
                FROM iceberg_nc4cloudprc_hadoop.tsp_ods.ods_micar_tsp_bms_file_sampled_data
                WHERE date >= '{date1}' and DATE <= '{date2}')
            SELECT
                A.*,
                B.hvbattcellmaxsocerr
            FROM A 
            JOIN B 
            ON A.vin = B.vid and round(A.start_timestep/1000) = B.ts
            """,
    # 提取AI-SOC 原始充电（工况）数据
    'sql3' : f"""
            WITH A AS (
                SELECT
                    vin,
                    date,
                    start_timestep,
                    end_timestep,
                    miles,
                    meani,
                    start_tmin,
                    end_tmin
                FROM iceberg_nc4cloudprc_hadoop.bms.charge_feature
                WHERE configurationlevel='Modena-1' AND start_soc < 50 AND end_volmax > 3.78 AND hvbattchrgnmodsts > 1.5 AND meani < -50 
                AND date >= '{date1}' and date <= '{date2}' and vin = '{vin}'
                ),
            B AS (
                SELECT
                    vid,
                    split(pack_conf,"_")[0] as car_conf
                FROM iceberg_nc4cloudprc_hadoop.bms.dws_powertrain_baseinfo_df
                WHERE split(pack_conf,"_")[0] = 'Modena-1' 
                AND date >= '{date1}' and date <= '{date2}' and vid = '{vin}'
            ),
            C as (
                SELECT
                    A.*,
                    car_conf
                FROM A 
                JOIN B 
                ON A.vin = B.vid ),
            D as (
                SELECT
                    vin,
                    date,
                    businesstimestamp,
                    hvbattactcur,
                    hvbattcellmaxsoc, 
                    hvbattcellminsoc, 
                    hvbattucellmax, 
                    hvbattucellmin,
                    hvbattcelltmax, 
                    hvbattcelltmin,
                    totodoacrt,
                    hvbattcelltmax,
                    hvbattcelltmin,
                    (hvbattcelltmax + hvbattcelltmin)/2 as hvbattcelltavg,
                    (-1 * hvbattactcur) as curr
                FROM
                    iceberg_nc4cloudprc_hadoop.bms.dwd_micar_tsp_vccd_cycle_extract_di_common
                where date >= '{date1}' and date <= '{date2}' and vin = '{vin}'
                ),
            E as (
                SELECT
                    vin,
                    date,
                    gcsoh_offlinetotaldecay_ah_alldata/100 as gcsoh_offlinetotaldecay_ah_alldata
                FROM iceberg_nc4cloudprc_hadoop.bms.dws_powertrain_bms_packinfo_part_4_soh_1d
                WHERE date >= '{date1}' and date <= '{date2}' and vin = '{vin}'
            )
            SELECT
                D.*,
                C.*,
                (188.8 - E.gcsoh_offlinetotaldecay_ah_alldata)/188.8*100 as offlinesoh
            FROM D
            JOIN C
            ON D.vin = C.vin and D.businesstimestamp >= C.start_timestep 
            and D.businesstimestamp <= C.end_timestep
            JOIN E
            ON D.vin = E.vin
    """}
    print(sql_type.get(sql))
    result_df = select_pyhive(sql_type.get(sql), hostname, username)
    # result_df.to_csv(savepath+'1.csv')
    return result_df


if __name__ == "__main__":
    # date1_list = [20250116, 20250205, 20250204, 20250205, 20250202, 20250203, 20250201, 20250202, 20250201, 20250205, 20250131,
    #               20250116, 20250120, 20250117, 20250127, 20250129, 20250129, 20250119, 20250131, 20250126, 20250124, 20250130]# 起始日期
    # date2_list = [20250116, 20250205, 20250204, 20250205, 20250202, 20250203, 20250201, 20250202, 20250201, 20250205, 20250131,
    #                   20250116, 20250120, 20250117, 20250127, 20250129, 20250129, 20250119, 20250131, 20250126, 20250124, 20250130]# 结束日期
    # vin_list = ["HXMQED3CLV3N9UVY1","LNBQY3JFH0RF9BA58", "LNBQJHFFEH7SLVAD1", "LNBQU6URBWFF5ZZC0","LNBQGC7BXLDUABJW9","LNBQMSNRY0GF3RBU3",
    #             "LNBQWYN5LA5XM7TD4", "LNBQL1ATPNNRBJM31", "HXMQA8XVXRLXZTTC2", "HXMQ1SUS6WXAUSF53", "LNBQ1WFVMNSEX1UC9",
    #             "LNBQEYSJB9NZL18Z8", "LNBQWGL56TRL0RSW4", "HXMQX3FCXZCPTH5Y7", "LNBQ90YY8E2TFGGX3", "LNBQPEWBTTRWAASG4",
    #             "LNBQ3XB2BLBRTJW54", "HXMQYCSLY8ZAKCUW7", "LNBQNVGGXNRVF4KY7", "LNBQLKEBFGMVXLHC4", "HXMQN0GLVA1LGYDV1",
    #             "HXMQXLRJX8SC3VL92"]
    date1_list = [20250125, 20250127, 20250205]# 起始日期
    # date2_list = [20250125, 20250205, 20250204, 20250205]# 结束日期
    vin_list = ["LNBQZW859FHSPEU90","LNBQDNDVN0HT5FN91", "LNBQZ9YPPKFLDXBP0"]

    # date1_list = [20250205, 20250123, 20250121, 20250123]# 起始日期
    # # date2_list = [20250125, 20250205, 20250204, 20250205]# 结束日期
    # vin_list = ["LNBQYKVMEHKH7UEX3","HXMQEU2BHJVBURNK6", "LNBQKCJVYPR6DCZA8", "LNBQLTRXTALD5G0M3"]

    savepath = 'D:/Code/code/submit/bms_ai/PycharmProjects/data/badcase_csv_0211/'    #数据保存路径

    for i in range(len(date1_list)):
        date1 = date1_list[i]
        date2 = date1_list[i]
        vin = vin_list[i]
        #  拉取数据时，根据需求更改以上信息，数据保存路径及文件名可根据需求更改
        #  sql1：拉取结果表数据，可规定日期范围
        #  sql2：拉取结果表数据，可规定日期范围
        #  sql3：拉取单车原始数据，可规定日期范围（单天为一个充电片段，若日期范围非单天则为多个充电片段）
        data = maincode('sql3', date1, date2, vin, savepath)
        data.to_csv(f"{savepath}{vin}.csv", index=False)
        # print(data)




