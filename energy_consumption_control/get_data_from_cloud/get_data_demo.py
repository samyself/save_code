#!/usr/bin/env python3

import http.client
import sys
import json
import time


def get_json(path, body=None):
    conn = http.client.HTTPConnection(
        "proxy-service-http-nc4cloudprc-dp.api.xiaomi.net")

    headers = {
        'x-sqlproxy-user': "deb873fc42b34b3cbfa3da9af22908ab",
        'x-sqlproxy-engine': "presto",
        'x-sqlproxy-catalog': "iceberg_nc4cloudprc_hadoop",
        'content-type': "text/plain;charset=utf-8",
    }

    conn.request("POST", path, body, headers)

    res = conn.getresponse()

    if res.status != 200:
        fail(res.reason)

    data = res.read().decode("utf-8")
    return json.loads(data)


def submit_sql(sql):
    path = "/olap/api/v2/statement/query"
    payload = sql.encode("utf-8")
    query_id = get_json(path, payload).get("data").get("queryId")
    return query_id


def get_state(query_id):
    path = "/olap/api/v2/statement/getStatusAndLog"
    path = path + "?queryId=" + query_id
    data = get_json(path).get("data")
    print(data)
    state = data.get("state")
    query_id = data.get("nextQueryId")
    return (state, query_id)


def get_result(query_id):
    path = "/olap/api/v2/statement/fetchResult"
    next_query_id = query_id
    while True:
        (state, next_query_id) = get_state(next_query_id)
        print("current state: " + state)
        if state == "FINISHED":
            break
        elif state == "FAILED":
            fail("query failed")
        else:
            time.sleep(5)
            pass
    url = path + "?queryId=" + query_id
    result = get_json(url)
    out_data={ }
    while True:
        if result.get("data").get("nextResultQueryId") != '':
            next_query_id = result.get("data").get("nextResultQueryId")
            url = path + "?queryId=" + next_query_id
            result = get_json(url)
            columns = result.get("data").get("columns")
            rows=result.get("data").get("rows")
            for data_index in range(len(rows)):
                now_row=rows[data_index]
                for name_data_index in range(len(columns)):
                    name_data=columns[name_data_index]
                    now_data=now_row[name_data_index]
                    name=name_data['name']
                    if name in out_data:
                        out_data[name].append(now_data)
                    else:
                        out_data[name]=[now_data]
            # print("columns:")
            # print(result.get("data").get("columns"))
            # print("rows:")
            # print(result.get("data").get("rows"))

        else:
            # print("columns:")
            # print(result.get("data").get("columns"))
            # print("rows:")
            # print(result.get("data").get("rows"))
            for data_index in range(len(rows)):
                now_row=rows[data_index]
                for name_data_index in range(len(columns)):
                    name_data=columns[name_data_index]
                    now_data=now_row[name_data_index]
                    name=name_data['name']
                    if name in out_data:
                        out_data[name].append(now_data)
                    else:
                        out_data[name]=[now_data]
            print("get result finished")
            break
    return out_data


def query(sql):
    query_id = submit_sql(sql)
    out_data = get_result(query_id)
    return out_data


def fail(msg):
    print(msg)
    sys.exit(-1)

# for test
# ./kyuubi-python-request.py "select a, b, c from values('a', 'b', 1),('aa', 'bb', 2) as t(a, b, c)"

if __name__ == "__main__":

    # sql = "SELECT temp, press FROM iceberg_nc4cloudprc_hadoop.tmp.r134a_temp_at_sat_versus_press limit 1;"
    try_id = 0
    while True:
        try:
            sql = "SELECT * FROM iceberg_nc4cloudprc_hadoop.bms.ods_micar_tsp_tms_file_sampled_data_backup limit 1000;"
            out_data = query(sql)
            break
        except Exception as e:
            print(e)
            print('retry...',try_id)
            time.sleep(10)
            continue
    print(out_data)
    sys.exit(0)
