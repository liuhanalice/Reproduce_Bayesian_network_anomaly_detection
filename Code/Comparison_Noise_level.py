import numpy as np
import matplotlib.pyplot as plt


def figure_noise_level():
    name = ['EM_DRG', 'J_Nondestruct', 'IST_OPTIC', 'EFA','GRAPHICAL_MODEL']
    noise_level = ['0p05', '0p10', '0p15']
    for i in range(len(noise_level)):
        print('\n')
        print("noise level:", noise_level[i])
        file = './EM_DRG_result/Noise_' + noise_level[i]
        if i == 0:
            len_name = 5
        else:
            len_name = 4
        for j in range(len_name):
            print("name:", name[j], end='   ')
            file_FPR = file + '/result3/' + name[j] + '_FPR.npy'
            file_FNR = file + '/result3/' + name[j] + '_FNR.npy'
            FPR = np.load(file_FPR).mean()
            FNR = np.load(file_FNR).mean()
            print("FPR and FNR:", FPR, FNR, end='   ')


if __name__ == '__main__':
    figure_noise_level()
