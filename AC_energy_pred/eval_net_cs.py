import os
from mpl_toolkits.mplot3d import Axes3D
import torch
from matplotlib import pyplot as plt

from model import *

if __name__ == '__main__':
    status_model = StatusPredModel(model_dict=model_dict,
                                 sub_status_model_name_list=sub_status_model_name_list,
                                 para_dict=para_dict,
                                 model_input_name_dict=model_input_name_dict,
                                 model_output_name_dict=model_output_name_dict)
    status_model.load_sub_model(sub_model_ckpt_path_dict)
    status_model.eval()
    model = {}
    #model['status_model'] = status_model

    base_path = sub_model_folder
    lp_model = LPModel(**para_dict['LPModel'])
    model_path = os.path.join(base_path, model_ckpt_name_dict['LPModel'])
    lp_model.load_state_dict(torch.load(model_path))
    lp_model.eval()
    model = {}
    model['LPModel'] = lp_model

    pic_path_folder = './pic/net_out'
    if not os.path.exists(pic_path_folder):
        os.makedirs(pic_path_folder)

    with torch.no_grad():
        for net_name in model.keys():
            net = model[net_name]
            if net_name == 'heat_power_net':
                current = torch.range(0, 1, 0.050)
                heat_power_weights = net(current.view(-1, 1)).view(-1)
                plt.plot(current, heat_power_weights, label=net_name)

            else:

                ags = torch.arange(0, 90, 6)
                fan = torch.arange(0, 100, 1)

                ags_array = ags.view(1,-1).repeat(len(fan),1).view(-1)
                fan_array = fan.view(-1,1).repeat(1,len(ags)).view(-1)
                weight_input=torch.zeros((len(ags)*len(fan), len(model_input_name_dict[net_name])))

                ## 特征数据替换
                fea_data_dict = { "hi_pressure": 567.666849,
        "temp_p_h_5": 10.784127,
        "compressor_speed": 1000,
        "cab_heating_status_act_pos": 90,
        "ags_openness": ags_array, #需要变化的值
        "cfan_pwm": fan_array, #需要变化的值
        "car_speed": 60,
        "temp_amb": -14.75}

                for i in range(len(model_input_name_dict[net_name])):
                    fea_name = model_input_name_dict[net_name][i]
                    weight_input[:, i] = fea_data_dict[fea_name]


                # weight_input = torch.cat([torch.abs(speed).view(-1, 1), flowrate_oil.view(-1, 1)], dim=-1)
                state = torch.FloatTensor(weight_input)
                output_data = model[net_name](state)

                # 开始画图
                lo_pressure = output_data[:, 0].flatten()
                temp_p_h_1_cab_heating = output_data[:, 1].flatten()
                print(len(lo_pressure), len(temp_p_h_1_cab_heating), len(weight_input[:, 4]), len(weight_input[:, 5]))
                print(lo_pressure, temp_p_h_1_cab_heating)
                fig = plt.figure()  # 创建一个画布figure，然后在这个画布上加各种元素。
                ax = fig.add_subplot(121, projection='3d')  # 创建3D坐标轴
                scatter = ax.scatter(weight_input[:,4], weight_input[:,5], lo_pressure, c=lo_pressure, cmap='viridis')
                # 添加颜色条
                fig.colorbar(scatter, ax=ax, label='lo_pressure')
                ax.set_xlabel('ags')
                ax.set_ylabel('fan')
                ax.set_zlabel('lo_pressure')
                fig.suptitle(f"hi_pressure={fea_data_dict['hi_pressure']}")

                ax2 = fig.add_subplot(122, projection='3d')  # 创建3D坐标轴
                scatter2 = ax2.scatter(weight_input[:, 4], weight_input[:,5], temp_p_h_1_cab_heating, c=temp_p_h_1_cab_heating, cmap='viridis')
                # 添加颜色条
                fig.colorbar(scatter2, ax=ax2, label='temp_p_h_1_cab_heating')
                ax2.set_xlabel('ags')
                ax2.set_ylabel('fan')
                ax2.set_zlabel('temp_p_h_1_cab_heating')
                #ax2.set_title(f"low_presure={temp_amb}, sh_act_mode_10={sh_temp}, \n"
                #             f"sh_tar_mode_10={tar_sh_temp}, last_ags={last_ags}, last_fan={last_fan}")
                fig.tight_layout()
                plt.show()

            pic_path = os.path.join(pic_path_folder, f'{net_name}_data.png')
            plt.savefig(pic_path)
            plt.close()
