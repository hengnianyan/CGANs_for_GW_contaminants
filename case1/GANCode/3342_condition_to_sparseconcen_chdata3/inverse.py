import os
# import sys
import pickle
import numpy as np
import tensorflow as tf
# import PIL.Image
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset
import matplotlib as mpl

np.random.seed(1000)

tfutil.init_tf(config.tf_config)



current_dir = os.getcwd()
os.environ['CUDA_VISIBLE_DEVICES']="3" # only one gpu is visible

data_dir_test = '/public/home/yhn/back3_test7/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell/000-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'


test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_3342_chdata3_lesswell', max_label_size = 'full')


N_test = 100
N_test2 = 100
reals = np.zeros([N_test2] + test_set.shape, dtype=np.float32)
label_test = np.zeros([N_test, test_set.label_size], dtype=np.float32)

realrefers = np.zeros([N_test] + [1,64,64], dtype=np.float32)

for idx in range(N_test):
    reals_t, label_t = test_set.get_minibatch_imageandlabel_np(1)  
    _, realrefers_t  = test_set.get_minibatch_well_np(1)
    
    reals[idx] = reals_t[0]
    label_test[idx] = label_t[0]

    realrefers[idx] = realrefers_t[0]



################################################### run model

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)



# time label
plt.rc('font',family='Times New Roman') #全局字体
fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')

fig.set_size_inches(10, 3.5, forward=True)
fig.dpi = 600
ftsize = 14
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(0.2 * 50) # 每隔整数取tick



abc_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']





_,time_est = D2.run(reals[0:N_test2]*2-1,minibatch_size=16)



print(label_test.shape)
print(time_est.shape)
for i in range(3):
    ax[i].scatter(label_test[0:N_test2,i], time_est[0:N_test2,i], s=40, c=plt.cm.Set1(1))
    # calc the trendline
    z = np.polyfit(label_test[0:N_test2,i], time_est[0:N_test2,i], 1)
    p = np.poly1d(z)
    min_plt = np.min(label_test[0:N_test2,i])+0.1
    max_plt = np.max(label_test[0:N_test2,i])-0.1

    ax[i].plot(np.array([min_plt,max_plt]),p(np.array([min_plt,max_plt])), color=plt.cm.Set1(0), linewidth=3)

    y2 = z[0] * label_test[0:N_test2,i] + z[1]
    correlation = np.corrcoef(time_est[0:N_test2,i], y2)[0,1]  #相关系数
    squareR = (correlation**2)   #R方

    # the line equation:
    if z[1]<0:
        ax[i].text(0.1, 0.8,"$Y = %.2fX %.3f$"%(z[0],z[1]), fontsize=ftsize)
    else:
        ax[i].text(0.1, 0.8,"$Y = %.2fX + %.3f$"%(z[0],z[1]), fontsize=ftsize)
    ax[i].text(0.1, 0.7,"$R^2 = %.3f$"%(squareR), fontsize=ftsize)
    print ("y=%.6fx+(%.6f)"%(z[0],z[1]))


    # xy坐标轴设置
    ax[i].set_xlim(0, 1)
    ax[i].set_ylim(0, 1)

    ax[i].set_xticks(np.arange(0, 1.2, 0.2)) # 将行列号[0,1,2,...63]显示为距离m
    ax[i].set_xticklabels(np.round(np.arange(0, 1.2, 0.2), 1).tolist(), fontsize = ftsize)
    ax[i].set_yticks(np.arange(0, 1.2, 0.2))
    ax[i].set_yticklabels(np.round(np.arange(0, 1.2, 0.2), 1).tolist(), fontsize = ftsize)

    ax[i].set_xlabel("Reference parameters", fontsize=ftsize)
    ax[i].set_title('('+ abc_list[i] + ')', fontsize=ftsize)
    

    ax[i].set_aspect('equal', adjustable='box')

ax[0].set_ylabel("Estimated parameters", fontsize=ftsize)
plt.tight_layout()
plt.savefig(network_dir +"Time step fake vs real2.png", dpi=600)
