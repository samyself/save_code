from MICAN_Control.CanfdCtrl import CanfdCtrl
from MICAN_Control.canFrame import canFrame
from threading import Thread
import time, sys, signal
import pandas as pd

def signal_recv():
    while 1:
        msg = canfdctrl.receive_data()
        if msg:
            if msg.arbitration_id == 849 and msg.channel == 1:
                signal_val_0 = canfdctrl.get_signal(msg, "ImpctEvntSts")
                signal_val_1 = canfdctrl.get_signal(msg, "ImpctEvntStsSigGrpCntr")
                print("signal val 0 1 :", signal_val_1, signal_val_0)

        else:
            time.sleep(1)



if __name__ == "__main__":
    df_origin = pd.read_csv('D:/Code/code/submit/bms_ai/PycharmProjects/data/Graphics_test.csv')
    # 前向填充
    ffill_df = df_origin.ffill()
    # 后向填充
    bfill_df = df_origin.bfill()
    # 组合填充
    df1 = ffill_df.combine_first(bfill_df)


    # 打开设备
    canfdctrl = CanfdCtrl(open_device=True)
    # 获取设备软件版本号
    version = canfdctrl.get_device_version
    print(f"当前设备软件版本号为{version}")

    # 加载DBC配置,为can设备的1路配置ZCU的DBC
    canfdctrl.loadDBC(1, r"D:/Code/code/submit/bms_ai/PycharmProjects/data/MS11_PTCANFD_230803.dbc", decode="gbk")

    # 开启线程查看can报文接收情况
    rece_thd = Thread(target=signal_recv, daemon=True)
    rece_thd.start()
    # 配置can单帧发送报文
    for i in range(len(df1['HvBattActCur'])):
        start_time,end_time=time.time(),time.time()
        while(end_time-start_time<0.05):
            config_send_list=[canFrame(
            BUSId=1,
            CANId=145,
            data={"HvBattActCur":df1['HvBattActCur'][i]},
            slot=1),
            canFrame(
            BUSId=1,
            CANId=1010,
            data={"HvBattCellTMin":df1['HvBattCellTMin'][i],"HvBattCellTMax":df1['HvBattCellTMax'][i]},
            slot=2),
            canFrame(
            BUSId=1,
            CANId=912,
            data=[df1['HvBattCellMaxSoc'][i]],
            slot=3),
            canFrame(
            BUSId=1,
            CANId=148,
            data=[df1['HvBattWorkSts'][i]],
            slot=4),
            canFrame(
            BUSId=1,
            CANId=630,
            data=[df1['HvBattUCellMax'][i]],
            slot=5),
            ]
            # 发送can单帧报文
            for config_send_once in config_send_list:
                canfdctrl.send_once(config_send_once)
            end_time=time.time()


