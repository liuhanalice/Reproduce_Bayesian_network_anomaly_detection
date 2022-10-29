import matplotlib.pyplot as plt
import numpy as np

from my_funcs import *
import matplotlib as mpl


def c_u_FPR(FPR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    cf_list = [1 / 16, 1 / 12, 1 / 10, 1 / 8, 1/6, 1 / 4, 1 / 2]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FPR_vec, label='FPR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$C_u$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)
    char_cf_list = [r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$']
    cf_list_round = np.round(np.array(cf_list), 3)
    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(cf_list)), cf_list_round)
    plt.xlim(-0.1, len(cf_list) - 1 + 0.2)
    plt.hlines(np.max(FPR_vec), -0.1, np.argmax(FPR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.2, 9))
    plt.savefig('./figures/c_u_FPR.jpg', dpi=300)
    plt.show()


def c_u_FNR(FNR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    cf_list = [1 / 16, 1 / 12, 1 / 10, 1 / 8, 1/6, 1 / 4, 1 / 2]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FNR_vec, label='FNR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$C_u$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FNR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)
    char_cf_list = [r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$', r'$C_u$']
    cf_list_round = np.round(np.array(cf_list), 3)
    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(cf_list)), cf_list_round)
    plt.xlim(-0.1, len(cf_list) - 1 + 0.2)
    plt.hlines(np.min(FNR_vec), -0.1, np.argmin(FNR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.48, 9))
    plt.savefig('./figures/c_u_FNR.jpg', dpi=300)
    plt.show()



def sensitivity_study_c_u():
    cf_list = [1 / 16, 1 / 12, 1 / 10, 1 / 8, 1/6, 1 / 4, 1 / 2]
    FPR_vec = np.zeros((len(cf_list),))
    FNR_vec = np.zeros((len(cf_list),))
    for i in range(len(cf_list)):
        cf = cf_list[i]
        confusion_mat = np.load('./parameter sensitivity/cf_sensitivity_' + str(cf) + '.npy')
        FPR = (confusion_mat[0, 1] + confusion_mat[2, 1]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[2, :]))
        FNR = (confusion_mat[1, 0] + confusion_mat[1, 2]) / np.sum(confusion_mat[1, :])
        FPR_vec[i] = FPR
        FNR_vec[i] = FNR
    print("FPR_vec:", FPR_vec)
    print("FNR_vec:", FNR_vec)
    c_u_FPR(FPR_vec)
    c_u_FNR(FNR_vec)
    # c_u(FPR_vec,FNR_vec)


def n_knot_FPR(FPR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    n_knot_list = [9, 10, 11, 12, 13, 14, 15, 16]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FPR_vec, label='FPR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$n_{knot}$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)

    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(n_knot_list)), n_knot_list)
    plt.xlim(-0.1, len(n_knot_list) - 1 + 0.2)
    # plt.hlines(np.max(FPR_vec), -0.1, np.argmax(FPR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.2, 9))
    plt.savefig('./figures/n_knot_FPR.jpg', dpi=300)
    plt.show()


def n_knot_FNR(FNR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    n_knot_list = [9, 10, 11, 12, 13, 14, 15, 16]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FNR_vec, label='FNR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$n_{knot}$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FNR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)
    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(n_knot_list)), n_knot_list)
    plt.xlim(-0.1, len(n_knot_list) - 1 + 0.2)
    plt.hlines(np.max(FNR_vec), -0.1, np.argmax(FNR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.48, 9))
    plt.savefig('./figures/n_knot_FNR.jpg', dpi=300)
    plt.show()


def sensitivity_study_n_knot():
    n_knot_list = [9, 10, 11, 12, 13, 14, 15, 16]
    FPR_vec = np.zeros((len(n_knot_list),))
    FNR_vec = np.zeros((len(n_knot_list),))
    for i in range(len(n_knot_list)):
        n_knot = n_knot_list[i]
        confusion_mat = np.load('./parameter sensitivity/num_knot_sensitivity_' + str(n_knot) + '.npy')
        FPR = (confusion_mat[0, 1] + confusion_mat[2, 1]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[2, :]))
        FNR = (confusion_mat[1, 0] + confusion_mat[1, 2]) / np.sum(confusion_mat[1, :])
        FPR_vec[i] = FPR
        FNR_vec[i] = FNR
    print("FPR_vec:", FPR_vec)
    print("FNR_vec:", FNR_vec)
    n_knot_FPR(FPR_vec)
    n_knot_FNR(FNR_vec)


def k_FPR(FPR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    k_list = [25, 30, 35, 40, 45, 50]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FPR_vec, label='FPR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$k$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)

    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(k_list)), k_list)
    plt.xlim(-0.1, len(k_list) - 1 + 0.2)
    # plt.hlines(np.max(FPR_vec), -0.1, np.argmax(FPR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.2, 9))
    plt.savefig('./figures/k_FPR.jpg', dpi=300)
    plt.show()


def k_FNR(FNR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    k_list = [25, 30, 35, 40, 45, 50]
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FNR_vec, label='FNR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$k$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FNR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)
    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(k_list)), k_list)
    plt.xlim(-0.1, len(k_list) - 1 + 0.2)
    # plt.hlines(np.max(FNR_vec), -0.1, np.argmax(FNR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.48, 9))
    plt.savefig('./figures/k_FNR.jpg', dpi=300)
    plt.show()


def sensitivity_study_k():
    k_list = [25, 30, 35, 40, 45, 50]
    FPR_vec = np.zeros((len(k_list),))
    FNR_vec = np.zeros((len(k_list),))
    for i in range(len(k_list)):
        k = k_list[i]
        confusion_mat = np.load('./parameter sensitivity/k_sensitivity_' + str(k) + '.npy')
        FPR = (confusion_mat[0, 1] + confusion_mat[2, 1]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[2, :]))
        FNR = (confusion_mat[1, 0] + confusion_mat[1, 2]) / np.sum(confusion_mat[1, :])
        FPR_vec[i] = FPR
        FNR_vec[i] = FNR
    print("FPR_vec:", FPR_vec)
    print("FNR_vec:", FNR_vec)
    k_FPR(FPR_vec)
    k_FNR(FNR_vec)


def c_ns_FPR(FPR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    c_ns_list = [1 / 24, 1 / 36, 1 / 48, 1 / 64, 1 / 80, 1 / 120]
    c_ns_list = list(np.sort(np.array(c_ns_list)))
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FPR_vec, label='FPR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$C_{ns}$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FPR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)

    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(c_ns_list)), np.round(c_ns_list, 3))
    plt.xlim(-0.1, len(c_ns_list) - 1 + 0.2)
    # plt.hlines(np.max(FPR_vec), -0.1, np.argmax(FPR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.2, 9))
    plt.savefig('./figures/c_ns_FPR.jpg', dpi=300)
    plt.show()


def c_ns_FNR(FNR_vec):
    from matplotlib.pyplot import figure
    figure(figsize=(8, 6), dpi=80)
    c_ns_list = [1 / 24, 1 / 36, 1 / 48, 1 / 64, 1 / 80, 1 / 120]
    c_ns_list = list(np.sort(np.array(c_ns_list)))
    rc = {"font.family": "serif",
          "mathtext.fontset": "cm"}
    plt.rcParams.update(rc)
    plt.xticks(fontproperties='Times New Roman', size=26)
    plt.yticks(fontproperties='Times New Roman', size=26)
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams['figure.figsize'] = (50.0, 8.0)
    plt.plot(FNR_vec, label='FNR', marker='*', color='blue', markersize=10)
    plt.xlabel(r'$C_{ns}$', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylabel('FNR', fontdict={'family': 'Times New Roman', 'size': 32})
    plt.ylim([-0.01, 0.2])
    plt.subplots_adjust(bottom=0.20, left=0.20)
    # plt.xticks(np.arange(len(cf_list)), [r'$^{1}/_{96}$', r'$^{1}/_{48}$', r'$^{1}/_{24}$', r'$^{1}/_{12}$', r'$^{1}/_{8}$', r'$^{1}/_{4}$', '$^{1}/_{2}$', ],
    #            math_fontfamily='cm')
    plt.xticks(np.arange(len(c_ns_list)), np.round(c_ns_list, 3))
    plt.xlim(-0.1, len(c_ns_list) - 1 + 0.2)
    # plt.hlines(np.min(FNR_vec), -0.1, np.argmin(FNR_vec), linestyles='--', color='black')
    plt.yticks(np.linspace(0, 0.48, 9))
    plt.savefig('./figures/C_ns_FNR.jpg', dpi=300)
    plt.show()


def sensitivity_study_c_ns():
    c_ns_list = [1 / 24, 1 / 36, 1 / 48, 1 / 64, 1 / 80,  1 / 120]
    c_ns_list = list(np.sort(np.array(c_ns_list)))
    FPR_vec = np.zeros((len(c_ns_list),))
    FNR_vec = np.zeros((len(c_ns_list),))
    for i in range(len(c_ns_list)):
        confusion_mat = np.load('./parameter sensitivity/c_ns_sensitivity_' + str(np.round(c_ns_list[i], 4)) + '.npy')
        FPR = (confusion_mat[0, 1] + confusion_mat[2, 1]) / (
                np.sum(confusion_mat[0, :]) + np.sum(confusion_mat[2, :]))
        FNR = (confusion_mat[1, 0] + confusion_mat[1, 2]) / np.sum(confusion_mat[1, :])
        FPR_vec[i] = FPR
        FNR_vec[i] = FNR
    print("FPR_vec:", FPR_vec)
    print("FNR_vec:", FNR_vec)
    c_ns_FPR(FPR_vec)
    c_ns_FNR(FNR_vec)

if __name__ == '__main__':
    # sensitivity_study_c_u()
    # sensitivity_study_n_knot()
    # sensitivity_study_k()
    sensitivity_study_c_ns()
