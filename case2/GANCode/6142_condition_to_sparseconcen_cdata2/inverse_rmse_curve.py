import os
import pickle
import numpy as np
import tensorflow as tf
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset

current_dir = os.getcwd()



data_dir_test = '/public/home/yhn/back6_test10/tfdataset/'


#trained generator directory path;
network_dir = '/public/home/yhn/back6_test10/GANresults/TrainingResults_6142_cdata2_lesswell/007-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'
results_dir = 'optimal_exp2_results_refer6_Nexp40_Cprior_wdlabel_N_vobs2000_reverseKL_rematch/'

rmse = np.loadtxt(network_dir + results_dir + 'rmse.txt')


plt.rc('font',family='Times New Roman') #全局字体
plt.rc('mathtext',fontset='stix') # 公式字体

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(5, 5, forward=True)
fig.dpi = 600
ftsize = 12

colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'black', 'tab:red','tab:cyan']
marks = ['-o','-*','-^','-x','-+','-v','-D','-s','-1']

step = 40

s = []
for i in range(8):
    s_temp, = ax.plot(np.arange(1,step+1+1), rmse[0:step+1,i], marks[i], color=colors[i], linewidth=2.5, markersize=5)

    s.append(s_temp)

ax.set_xlabel("Steps", fontsize = ftsize)
ax.set_ylabel("RMSE", fontsize = ftsize)

ax.legend(handles=s, \
        labels=['Release time', '$x_2$ for $s_1$', '$x_1$ for $s_1$', '$x_2$ for $s_2$', '$x_1$ for $s_2$', '$x_2$ for $s_3$', '$x_1$ for $s_3$'], markerscale=1, fontsize=ftsize)


plt.savefig(network_dir +  results_dir + "RMSE-040_2.png" , dpi=600)