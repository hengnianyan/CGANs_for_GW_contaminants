import os
import pickle
import numpy as np
import matplotlib as mpl
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt


current_dir = os.getcwd()


data_dir_test = '/public/home/yhn/back3_test7/tfdataset/'



network_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell/000-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'

def read_txt(inputpath, outputpath):
    with open(outputpath, 'w', encoding='utf-8') as file:
        with open(inputpath, 'r', encoding='utf-8') as infile:

            # 每行分开读取
            line = infile.readline()             # 调用文件的 readline()方法 
            while line:
                if line[0:6] == 'D_loss':
                    print(line)                 
                    
                    str_split = line.split(' ', 6)
                    file.write(' '.join(str_split[1::2])) # 只保留值
                    # file.write('\r\n') # 换行
                    # file.write('\n') # 换行
                line = infile.readline()  



 
 
if __name__ == "__main__":
    input_path = network_dir + 'log.txt'
    output_path = network_dir + 'train_loss.txt'
    kimg_path = network_dir + 'train_kimg.txt'

    read_txt(input_path, output_path)

    loss_data = np.loadtxt(output_path)
    loss_kimg = np.loadtxt(kimg_path)

    plt.rc('font',family='Times New Roman') #全局字体
    plt.rc('mathtext',fontset='stix') # 公式字体
    # fig, ax = plt.subplots(1, 2, sharex='col', sharey='row')
    fig, ax = plt.subplots(1, 2)
    # fig.set_size_inches(10, 5, forward=True)
    fig.set_size_inches(6.5, 3, forward=True)
    fig.dpi = 600
    ftsize = 12


    s1, = ax[0].plot(loss_kimg[:], loss_data[:, 2], '-o', color='tab:blue', linewidth=2.5, markersize=5)
    # ax[0].plot(loss_kimg[:], loss_data[:, 1], '-o', color='tab:orange', linewidth=2.5, markersize=5)
    ax[0].set_xlabel('Number of training images (' + '$10^3$' + ')', fontsize = ftsize)
    ax[0].set_ylabel("Loss", fontsize = ftsize)

    # plt.xticks(fontsize=ftsize)
    # plt.yticks(fontsize=ftsize)

    ax[0].set_title('(a)')

    s2, = ax[1].plot(loss_kimg[:], loss_data[:, 1], '-o', color='tab:orange', linewidth=2.5, markersize=5)

    ax[1].set_xlabel('Number of training images (' + '$10^3$' + ')', fontsize = ftsize)
    # ax[1].set_ylabel("Loss", fontsize = ftsize)



    # plt.xticks(fontsize=ftsize)
    # plt.yticks(fontsize=ftsize)
    # ax.set_yscale('log') 

    ax[1].set_title('(b)')


    # #添加图例
    # ax.legend(handles=[s1, s2, s3], \
    #         labels=['Optimized', 'Random', 'Uniform'], markerscale=1, fontsize=ftsize)

    plt.tight_layout()
    plt.savefig(network_dir + "Loss_plot.png" , dpi=600)



