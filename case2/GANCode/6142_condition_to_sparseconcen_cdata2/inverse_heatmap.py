import os

import pickle
import numpy as np
import tensorflow as tf

from matplotlib import use
import matplotlib as mpl
from matplotlib.patches import Rectangle

use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib.ticker as mtick
from scipy.ndimage.filters import gaussian_filter


np.random.seed(1000)
os.environ['CUDA_VISIBLE_DEVICES'] = "1" # only one gpu is visible
tfutil.init_tf(config.tf_config)

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]

def samples_entropy(samples_p,samples_q,bins=50,range=[0,1]):
    # 样本需要进行过归一化处理[0,1]，方便计算概率


    # 概率直方图，cnt为计数值
    cnt_p,_ = np.histogram(samples_p,bins=bins,range=range)
    cnt_q,_ = np.histogram(samples_q,bins=bins,range=range)

    p_p = cnt_p / np.sum(cnt_p)
    p_q = cnt_q / np.sum(cnt_q)


    return np.sum(p_q * np.log((p_q / (p_p+0.0001)) + 0.0001))


def myplot(x, y, s, rangexy, bins=100):
    # s 为高斯滤波参数，越大heatmap越平滑
    # rangexy [[xmin, xmax], [ymin, ymax]].
    # heatmap, xedges, yedges = np.histogram2d(x, y, range=rangexy, bins=bins, density=True)
    heatmap, xedges, yedges = np.histogram2d(x, y, range=rangexy, bins=bins)
    heatmap = gaussian_filter(heatmap, sigma=s)

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    return heatmap.T, extent




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

refer_index = 6 # 参考值样本

test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_6131_cdata2_lesswell', max_label_size = 'full')
training_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TrainingData_6131_cdata2_lesswell', max_label_size = 'full')


# labels are from -1 to 1
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




################################################### run model / Monte Carlo
# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)





plt.rc('font',family='Times New Roman') #全局字体
plt.rc('mathtext',fontset='stix') # 公式字体
fig, ax = plt.subplots(1, 3, sharex='col', sharey='row')
fig.set_size_inches(10, 5, forward=True)
fig.dpi = 600
ftsize = 12
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(0.2) # 每隔整数取tick


N_exp = 40  # 实验次数/打井的批次,必须>=2
N_mc = 1    # z向量的个数
N_virtual_obs = 2000 # 某一个井虚拟观测的个数,必须>=batch size
scale_size = 64
min_terval = 4
rows_list = np.arange(1, scale_size-1, min_terval)

cols_list = np.arange(26, scale_size-1, min_terval)
N_well_row = rows_list.shape[0]
N_well_col = cols_list.shape[0]


########### inversion
latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1])
well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')

optimal_well = np.loadtxt(network_dir + results_dir + "optimal_well.txt", dtype='int')

for step2 in range(N_exp):
    well_facies_temp[:,0,optimal_well[step2,0],optimal_well[step2,1]] = 1
    well_facies_temp[:,1,optimal_well[step2,0],optimal_well[step2,1]] = reals[refer_index,1,optimal_well[step2,0],optimal_well[step2,1]]

fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
_,label_est = D2.run(fake_images_out,minibatch_size=400)

label_est = label_est[:, [0, 1, 2, 5, 6, 3, 4]]


cmap_well = plt.cm.Reds
cmap_well.set_bad(color='white')
vmin = np.exp(0 * 6)
vmax = np.exp(0.7 * 6)
levels = 15

grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63


# 热图
sigma = 5

rangexy = np.array([[5, 5+20], [16, 16+30]], dtype='float32')
for i in range(3):
    # i = 2
    y = label_est[:, 1 + i*2]
    y = y * (47-17) + 17 - 1
    x = label_est[:, 2 + i*2]
    x = x * (26-6) + 6 - 1

    img, extent = myplot(x, y, sigma, rangexy)
    im = ax[i].imshow(img, extent=extent, origin='lower', cmap=cmap_well)


    cax = add_right_cax(ax[i], pad=0.002, width=0.01)
    cbar = fig.colorbar(im, cax=cax, ticks=locator, format=formatter)
    # cbar = fig.colorbar(im, cax=cax, format=formatter)
    cbar.ax.tick_params(labelsize=ftsize)


# 源
source_txt = [r'$s_1$', r'$s_2$', r'$s_3$']

for j in range(3): #子图

    for i in range(3):
        y = label_test[refer_index, 1 + i*2] * (47-17) + 17 - 1
        x = label_test[refer_index, 2 + i*2] * (26-6) + 6 - 1
        ax[j].scatter(x, y, s=70, c=plt.cm.Set1(5), marker='*', linewidths=1.5)
        ax[j].text(x+1, y+1, source_txt[i], fontsize=ftsize+2, color='black')


    # 方框和分割线
    ax[j].add_patch(Rectangle((5, 16), 20, 30, fill=False, edgecolor=plt.cm.Set1(1), linewidth=3, linestyle='--'))

    ax[j].axis([0,31,0,63])
    ax[j].set_aspect('equal', adjustable='box')
    ax[j].set_xticks(np.linspace(0, 30, 3)) # 将行列号[0,1,2,...63]显示为距离m
    ax[j].set_xticklabels(np.round(np.linspace(0, 46.875, 3), 0).astype(int).tolist(), fontsize = ftsize)
    ax[j].set_yticks(np.linspace(0, 60, 5))
    ax[j].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)
    ax[j].set_xlabel('${x_1}(m)$', fontsize = ftsize+2)
    
    if j == 0:
        ax[j].set_ylabel('${x_2}(m)$', fontsize = ftsize+2)



plt.savefig(network_dir + results_dir + "inverse_heatmap.png", dpi=600)

