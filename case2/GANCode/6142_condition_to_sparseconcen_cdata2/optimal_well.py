import os

import pickle
import numpy as np
import tensorflow as tf

from matplotlib import use
import matplotlib as mpl

use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import tfutil
import config
import dataset


np.random.seed(1000)
os.environ['CUDA_VISIBLE_DEVICES'] = "2" # only one gpu is visible
tfutil.init_tf(config.tf_config)


def add_right_cax(ax, pad, width):
    '''
    在一个ax右边追加与之等高的cax.
    pad是cax与ax的间距.
    width是cax的宽度.
    '''
    axpos = ax.get_position()
    caxpos = mpl.transforms.Bbox.from_extents(
        axpos.x1 + pad,
        axpos.y0,
        axpos.x1 + pad + width,
        axpos.y1
    )
    cax = ax.figure.add_axes(caxpos)

    return cax


current_dir = os.getcwd()



data_dir_test = '/public/home/yhn/back6_test10/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back6_test10/GANresults/TrainingResults_6142_cdata2_lesswell/007-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'
results_dir = 'optimal_exp2_results_refer6_Nexp40_Cprior_wdlabel_N_vobs2000_reverseKL_rematch/'


refer_index = 6 # 第几个样本为参考值

test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_6131_cdata2_lesswell', max_label_size = 'full')
training_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TrainingData_6131_cdata2_lesswell', max_label_size = 'full')


N_test = 50
N_test2 = 50
reals = np.zeros([N_test] + test_set.shape, dtype=np.float32)
label_test = np.zeros([N_test, test_set.label_size], dtype=np.float32)

realrefers = np.zeros([N_test] + [2,64,64], dtype=np.float32)

for idx in range(N_test):
    reals_t, label_t = test_set.get_minibatch_imageandlabel_np(1)  
    _, realrefers_t, _  = test_set.get_minibatch_well_np(1)

    reals[idx] = reals_t[0]
    label_test[idx] = label_t[0]

    realrefers[idx] = realrefers_t[0]



plt.rc('font',family='Times New Roman') #全局字体
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
fig.set_size_inches(6, 5, forward=True)
fig.dpi = 600
ftsize = 12
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(6) # 每隔整数取tick




##################### plot
N_exp = 40  # 实验次数/打井的批次,必须>=2
N_mc = 1    # z向量的个数
N_virtual_obs = 1000 # 某一个井虚拟观测的个数,必须>=batch size
scale_size = 64
min_terval = 4
rows_list = np.arange(1, scale_size-1, min_terval)
# cols_list = np.arange(29, scale_size-1, min_terval)
cols_list = np.arange(26, scale_size-1, min_terval)
N_well_row = rows_list.shape[0]
N_well_col = cols_list.shape[0]



######### concentration fields
cmap_well = plt.cm.GnBu
cmap_well.set_bad(color='white')
vmin = np.exp(0 * 6)
vmax = np.exp(0.7 * 6)
levels = 15

grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63

# 画两遍，避免有些地方是白色，且colorbar没有尖
im = ax.contourf(grid_X, grid_Y, np.exp(reals[refer_index,0,:,:] * 6),\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax.contourf(grid_X, grid_Y, np.exp(reals[refer_index,0,:,:] * 6),\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='both')



########### wells
optimal_well = np.loadtxt(network_dir + results_dir + "optimal_well.txt", dtype='int')

for i in range(N_exp):
    ##### optimal_well plot

    if i <= 9 :
        scatter_colors = 0
    else:
        scatter_colors = 8 # 灰色

    ax.scatter(optimal_well[i,1], 63-optimal_well[i,0],s=50,c=plt.cm.Set1(scatter_colors))
    
    if i <= 9:
        ax.text(optimal_well[i,1]+1,63-optimal_well[i,0]+1,str(int(i+1)),fontsize=ftsize+2)



# 源
source_txt = [r'$s_1$', r'$s_2$', r'$s_3$']
for i in range(3):
    y = label_test[refer_index, 1 + i*2] * (47-17) + 17 - 1
    x = label_test[refer_index, 2 + i*2] * (26-6) + 6 - 1
    ax.scatter(x, y, s=70, c=plt.cm.Set1(5), marker='*', linewidths=1.5)
    ax.text(x+1, y+1, source_txt[i], fontsize=ftsize+2, color='black')


# 方框和分割线
ax.add_patch(Rectangle((5, 16), 20, 30, fill=False, edgecolor=plt.cm.Set1(1), linewidth=3, linestyle='--'))

ax.axis([0,63,0,63])
ax.set_aspect('equal', adjustable='box')
ax.set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax.set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)
ax.set_yticks(np.linspace(0, 60, 5))
ax.set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


# colorbar设置
cax = add_right_cax(ax, pad=0.01, width=0.02)
cbar = fig.colorbar(im, cax=cax, ticks=locator, format=formatter)
cbar.ax.tick_params(labelsize=ftsize)



plt.savefig(network_dir + results_dir + "optimal_well.png", dpi=600)