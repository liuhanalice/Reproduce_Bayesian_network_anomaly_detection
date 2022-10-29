import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl


def figure_ratio():
    name = ['EM_DRG', 'J_Nondestruct', 'IST_OPTIC', 'EFA']
    ratio_level = ['1p4', '1p5', '1p6', '1p7', '1p8']
    FPR = {}
    FNR = {}
    len_name = len(name)
    for j in range(len_name):
        FPR[name[j]] = np.zeros((len(ratio_level),))
        FNR[name[j]] = np.zeros((len(ratio_level),))

    for i in range(len(ratio_level)):
        print("ratio level:", ratio_level[i])
        file = './EM_DRG_result/Ratio_' + ratio_level[i]

        for j in range(len_name):
            print("name:", name[j], end='   ')
            file_FPR = file + '/result3/' + name[j] + '_FPR.npy'
            file_FNR = file + '/result3/' + name[j] + '_FNR.npy'
            FPR[name[j]][i] = np.load(file_FPR).mean()
            FNR[name[j]][i] = np.load(file_FNR).mean()
    print('\n')
    print("FPR:", FPR)
    print("FNR:", FNR)
    ratio_FPR(FPR[name[2]], FPR[name[1]], FPR[name[0]],FPR[name[3]])
    ratio_FNR(FNR[name[2]], FNR[name[1]], FNR[name[0]],FNR[name[3]])


def ratio_FPR(IST_OPTIC_2008, J_Non_Eval_2017, EM_Spline,EFA):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 26})
    name_list = [r'Tang et al. (2009)', r'Jovanc$\check$evi$\acute{\mathrm{c}}$ et al (2017)', 'Our method', r'Miao et al. (2022)']
    color_list = ['red', 'blue', 'green', 'orange']
    plt.xlabel('Ratio ' + r'$\tau$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*']
    x_list = [1.4, 1.5, 1.6, 1.7, 1.8]
    y_list = [IST_OPTIC_2008, J_Non_Eval_2017, EM_Spline, EFA]
    for i in range(len(y_list)):
        plt.plot(x_list, y_list[i], color=color_list[i], linewidth=2, marker=marker_list[i], label=name_list[i])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=22, loc=1)
    plt.subplots_adjust(bottom=0.20, left=0.15)
    print(y_list)
    plt.ylim(-0.002, 0.26)
    plt.yticks([0.00,0.05,0.10,0.15,0.20,0.25])
    plt.xlim(1.39, 1.81)
    plt.savefig("FPR_ratio.jpg", dpi=300)
    plt.show()


def ratio_FNR(IST_OPTIC_2008, J_Non_Eval_2017, EM_Spline,EFA):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    mpl.rc('font', family='Times New Roman')
    rc = {"font.family": "serif",
          "mathtext.fontset": "stix"}
    plt.rcParams.update(rc)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.legend(prop={'family': 'Times New Roman', 'size': 26})
    name_list = [r'Tang et al. (2009)', r'Jovanc$\check$evi$\acute{\mathrm{c}}$ et al (2017)', 'Our method', r'Miao et al. (2022)']
    color_list = ['red', 'blue', 'green', 'orange']
    plt.xlabel('Ratio ' + r'$\tau$', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.ylabel('FNR', fontdict={'family': 'Times New Roman', 'size': 26})
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    marker_list = ['s', "o", '^', '*']
    x_list = [1.4, 1.5, 1.6, 1.7, 1.8]
    y_list = [IST_OPTIC_2008, J_Non_Eval_2017, EM_Spline, EFA]
    for i in range(len(y_list)):
        plt.plot(x_list, y_list[i], color=color_list[i], linewidth=2, marker=marker_list[i], label=name_list[i])
    plt.tick_params(labelsize=26)
    plt.legend(fontsize=22, loc=0)
    plt.subplots_adjust(bottom=0.20, left=0.15)
    plt.ylim(-0.05, 1.1)
    plt.yticks([0.00,0.25,0.50,0.75,1.00])
    plt.xlim(1.39, 1.81)
    plt.savefig("FNR_ratio.jpg", dpi=300)
    plt.show()


if __name__ == '__main__':
    figure_ratio()
