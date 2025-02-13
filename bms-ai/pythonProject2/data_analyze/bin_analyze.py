import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def calculate_gamma_bin(all_y_true, all_y_pred, all_gamma):
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_gamma = np.array(all_gamma)
    mae = np.mean(np.abs(all_y_true - all_y_pred))
    print("mae:", mae)
    # 统计不确定度小于1的个数

    # 调用分段绘图函数

    # 定义文件夹路径
    # folder_path = '../../data/pic/bin_analyse_Soc_1216_lstm_v1_h'  # 替换为实际路径
    #
    # # 检查文件夹是否存在，如果不存在则创建
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)
    #     print(f"Folder '{folder_path}' created.")
    # else:
    #     print(f"Folder '{folder_path}' already exists.")
    # plot_segments(all_y_true, all_y_pred, segment_size=200)
    # 设置pandas选项，以显示所有列
    # pd.set_option('display.max_columns', None)  # 显示所有列
    # pd.set_option('display.max_rows', None)  # 自动调整宽度
    # pd.set_option('display.max_colwidth', None)  # 显示完整宽度的字符串
    # print(all_y_pred)
    # print(all_y_true)
    # print(all_gamma)

    ####
    # percentiles_gamma = [i*10  for i in range(1, 10)]
    # bins_gamma = np.percentile(all_gamma, percentiles_gamma)
    # bins_gamma = np.insert(bins_gamma, 0, all_gamma.min())  # 在第一个元素之前插入一个元素
    # bins_gamma = np.append(bins_gamma, all_gamma.max())  # 在最后一个元素之后追加一个元素
    # # print(f'bins_gamma_num: {len(bins_gamma[0])}')
    # bins_gamma = np.unique(bins_gamma)
    # # Step 3: 创建分类器
    # labels = range(len(bins_gamma) - 1)
    # gamma_bins = pd.cut(all_gamma, bins=bins_gamma, labels=labels, include_lowest=True)
    # # Step 4: 计算 MAE
    # mae_by_bin = {}

    # for label in labels:
    #     mask = (gamma_bins == label)
    #     # mask = gamma_bins == label
    #     if mask.any():  # 确认有样本属于这个bin
    #         count = mask.sum()

    #         mae = np.mean(np.abs(all_y_true[mask] - all_y_pred[mask]))
    #         std = np.std(np.abs(all_y_true[mask] - all_y_pred[mask]))
    #         max = np.max(np.abs(all_y_true[mask] - all_y_pred[mask]))

    #         mask1 = (np.abs(all_y_true[mask] - all_y_pred[mask]) > 3*(np.exp(all_gamma[mask])**0.5))
    #         count_err = mask1.sum()

    #         mae_by_bin[label] = {
    #             'count': count,
    #             'count_err':count_err,
    #             'MAE': mae,
    #             'Std': std,
    #             'Max': max,
    #             'Range': f"[{bins_gamma[label]:.2f}, {bins_gamma[label + 1]:.2f})"
    #         }
    # # print('here')
    # # print('len mae_by_bin',len(mae_by_bin))
    # # 输出结果
    # for label, info in mae_by_bin.items():
    #     print(f"Bin {label}: Range = {info['Range']},Mae = {info['MAE']:.4f}, Std= {info['Std']:.4f}, Max= {info['Max']:.4f} ,Count = {info['count']:.4f},Count_err = {info['count_err']:.4f}")


if __name__ == '__main__':
    # 读取数据
    csv_path = '../../data/csv_res/Soc_0102_lstm_tcn_v1_3_delta_all'
    csv_files = os.listdir(csv_path)
    all_y_true = []
    all_y_pred = []
    all_gamma = []
    for file in csv_files:
        if '.csv' not in file:
            continue

        data = pd.read_csv(os.path.join(csv_path, file))

        # if (data[f'Curr'] <= 10.0).any():
        #     continue
        # if data[f'Curr'].values[0] < 100:
        #     continue

        y_true = data[f'Soc_real'].values
        y_pred = data[f'Soc_pred'].values
        gamma = data[f'unhealthy_value'].values
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_gamma.extend(gamma)

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_gamma = np.array(all_gamma)

    print(f'总点数为{len(all_y_true)}')
    Soc_list = [80.0, 85.0, 90.0, 95.0, 100.0]
    for i in range(len(Soc_list) - 1):
        # Soc在范围内
        mask = ((all_y_true >= Soc_list[i]) & (all_y_true < Soc_list[i + 1]))
        # print(mask)
        all_y_true_mask = all_y_true[mask]
        all_y_pred_mask = all_y_pred[mask]
        all_gamma_mask = all_gamma[mask]
        num_Soc_part = mask.sum()
        print(f'真实Soc的值从{Soc_list[i]}~{Soc_list[i + 1]}/n总共有{num_Soc_part}个点')
        if num_Soc_part <= 0:
            continue
        # calculate_gamma_bin(all_y_true, all_y_pred, all_gamma)

        # 不确定度小于1
        mask1 = all_gamma_mask <= 1.0
        num_Uncertainy_less1 = mask1.sum()
        print(f'不确定度小于1，总共有{num_Uncertainy_less1}个点')

        # 符合SOC在范围内且不确定度小于1的部分
        all_y_true_mask = all_y_true_mask[mask1]
        all_y_pred_mask = all_y_pred_mask[mask1]
        all_gamma_mask = all_gamma_mask[mask1]

        # 在剩余范围内挑选不符合3sigma原则的部分:
        # 理论上误差的均方根误差
        limit_max_err = np.sqrt(np.exp(all_gamma_mask))

        all_y_err_mask = np.abs(all_y_true_mask - all_y_pred_mask)
        mask2 = all_y_err_mask > 3 * limit_max_err
        num_uncertainy_err = mask2.sum()
        print(f'不确定度不符合预期的点的数量，总共有{num_uncertainy_err}个点,其中最大误差为{np.max(all_y_err_mask)}')

        print(f'****************************分割线******************************\n')

    # calculate_gamma_bin(all_y_true, all_y_pred, all_gamma)

    # mask_buqueding = all_gamma < 6.8
    # all_y_true = all_y_true[mask_buqueding]
    # all_y_pred = all_y_pred[mask_buqueding]
    # print('ss_mae = ', np.mean(np.abs(all_y_true - all_y_pred)))
    # # Step 2: 计算百分位数并创建分档
    # percentiles = [i * 10 for i in range(1, 10)]  # 10%, 20%, ..., 90%
    # bins = np.percentile(all_y_true, percentiles)
    # bins = np.insert(bins, 0, all_y_true.min())  # 在最前面添加最小值
    # bins = np.append(bins, all_y_true.max())  # 在最后面添加最大值
    # print(f'bins: {bins}')
    # bins = np.unique(bins)
    # # Step 3: 创建分档标签
    # labels = range(len(bins) - 1)
    # y_test_bins = pd.cut(all_y_true, bins=bins, labels=labels, include_lowest=True)
    #
    # # Step 4: 计算每个分档的 MAE
    # mae_by_bin = {}
    # for label in labels:
    #     mask = y_test_bins == label
    #     if mask.any():  # 确保该分档中存在数据
    #         mae = mean_absolute_error(all_y_true[mask], all_y_pred[mask])
    #         mae_by_bin[label] = {
    #             'MAE': mae,
    #             'Range': f"[{bins[label]:.2f}, {bins[label + 1]:.2f})"
    #         }
    #
    # # 输出结果
    # for label, info in mae_by_bin.items():
    #     print(f"Bin {label}: Range = {info['Range']}, MAE = {info['MAE']:.4f}")
    #
    # # 定义区间函数
    # def categorize_duration(duration):
    #     if duration < 20:
    #         return 0  # <20
    #     elif 20 <= duration <= 40:
    #         return 1  # 20-40
    #     else:
    #         return 2  # >40
    # # 将 all_y_true 和 all_y_pred 映射到区间
    # true_categories = np.array([categorize_duration(d) for d in all_y_true])
    # pred_categories = np.array([categorize_duration(d) for d in all_y_pred])
    #
    # # 计算准确率
    # accuracy = np.mean(true_categories == pred_categories)
    # print(f"Accuracy: {accuracy:.2%}")
    #
    # # 生成混淆矩阵
    # cm = confusion_matrix(true_categories, pred_categories, labels=[0, 1, 2])
    #
    # # 打印混淆矩阵
    # print("Confusion Matrix:")
    # print(cm)