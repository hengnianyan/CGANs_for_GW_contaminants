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
from matplotlib.ticker import MultipleLocator,FormatStrFormatter
import matplotlib.ticker as mtick

np.random.seed(1000)
os.environ['CUDA_VISIBLE_DEVICES'] = "0" # only one gpu is visible
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




current_dir = os.getcwd()


data_dir_test = '/public/home/yhn/back6_test10/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back6_test10/GANresults/TrainingResults_6142_cdata2_lesswell/007-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'
results_dir = 'optimal_exp2_results_refer6_Nexp40_Cprior_wdlabel_N_vobs2000_reverseKL_rematch/'

refer_index = 6 # 参考值样本

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




################################################### run model / Monte Carlo
# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)



N_exp = 40  # 实验次数/打井的批次,必须>=2
N_mc = 1    # z向量的个数
N_virtual_obs = 2000 # 某一个井虚拟观测的个数,必须>=batch size
scale_size = 64
min_terval = 4
rows_list = np.arange(1, scale_size-1, min_terval)
cols_list = np.arange(26, scale_size-1, min_terval)
N_well_row = rows_list.shape[0]
N_well_col = cols_list.shape[0]





wd = np.zeros([N_exp, scale_size, scale_size], dtype = 'float32')
optimal_well = np.zeros([N_exp, 2], dtype=int)
rmse = np.zeros([N_exp, 8], dtype = 'float32')
std = np.zeros([N_exp, 8], dtype = 'float32')

abe_prior = np.zeros([1, 7], dtype = 'float32')
abe_post = np.zeros([1, 7], dtype = 'float32')
rmse_prior = np.zeros([1, 7], dtype = 'float32')
rmse_post = np.zeros([1, 7], dtype = 'float32')
std_prior = np.zeros([1, 7], dtype = 'float32')
std_post = np.zeros([1, 7], dtype = 'float32')

_, realrefers_t, _  = training_set.get_minibatch_well_np(N_virtual_obs)  
realrefers_t = realrefers_t[:, 1:2]



#################### prior distribution of source parameters
label_test_prior = np.expand_dims(np.arange(0, 1, 1/(N_mc*N_virtual_obs)), axis = 1)
label_test_prior = np.repeat(label_test_prior, training_set.label_size, axis = 1)


latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1])
well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')



fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
_,label_est = D2.run(fake_images_out,minibatch_size=400)


label_est = label_est[:, [0, 1, 2, 5, 6, 3, 4]]

plt.rc('font',family='Times New Roman') #全局字体
fig, ax = plt.subplots(2, 4, sharex='col')
fig.set_size_inches(10, 5, forward=True)
fig.dpi = 600
ftsize = 12

abc_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

subplot_row = 0
subplot_col = 0
for i in range(7):

    if subplot_col >= 4:
        subplot_col = 0
        subplot_row += 1

    _, _, bar1= ax[subplot_row, subplot_col].hist(label_test_prior[:,i], range=(0,1), bins=50, \
            density=True, facecolor="grey", alpha=0.8)

    
    ax[subplot_row, subplot_col].set_xlim(0,1)

    abe_prior[0, i] = np.abs(label_test[refer_index,i] - label_test_prior[:,i].mean())
    std_prior[0, i] = np.std(label_test_prior[:,i])
    rmse_prior[0, i] = np.sqrt(((label_test[refer_index,i] - label_test_prior[:,i]) ** 2).mean())

    subplot_col += 1


######################### posterior  distribution
optimal_well = np.loadtxt(network_dir + results_dir + "optimal_well.txt", dtype='int')

for step2 in range(N_exp):
    well_facies_temp[:,0,optimal_well[step2,0],optimal_well[step2,1]] = 1
    well_facies_temp[:,1,optimal_well[step2,0],optimal_well[step2,1]] = reals[refer_index,1,optimal_well[step2,0],optimal_well[step2,1]]

fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
_,label_est = D2.run(fake_images_out,minibatch_size=400)

label_est = label_est[:, [0, 1, 2, 5, 6, 3, 4]]

subplot_row = 0
subplot_col = 0
for i in range(7):

    if subplot_col >= 4:
        subplot_col = 0
        subplot_row += 1

    hist_value, _, bar2= ax[subplot_row, subplot_col].hist(label_est[:,i], range=(0,1), bins=50, \
            density=True, facecolor=plt.cm.Set2(1), alpha=0.8)
    h = hist_value.max() * 0.7
    bar3, = ax[subplot_row, subplot_col].plot([label_test[refer_index,i],label_test[refer_index,i]] , [0,h],'r-',linewidth=3)
    
    


    ax[subplot_row, subplot_col].set_xlim(0,1)
    ax[subplot_row, subplot_col].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1f'))
    ax[subplot_row, subplot_col].grid(axis='y', ls='--', alpha=0.8)
    ax[subplot_row, subplot_col].set_title('('+ abc_list[i] + ')', fontsize=ftsize)
    
    abe_post[0, i] = np.abs(label_test[refer_index,i] - label_est[:,i].mean())
    std_post[0, i] = np.std(label_est[:,i])
    rmse_post[0, i] = np.sqrt(((label_test[refer_index,i] - label_est[:,i]) ** 2).mean())

    subplot_col += 1


ax[1, 2].legend(handles=[bar1[0], bar2[0], bar3], \
        labels=['Prior','Posterior', 'Reference'], markerscale=1.5, fontsize=ftsize, \
        bbox_to_anchor=(1.05, 0), loc=3, borderaxespad=0)

ax[1, 3].set_visible(False)  
plt.savefig(network_dir + results_dir + "inverse_hist2.png", dpi=600)



# 保存表格数据
np.savetxt(network_dir + results_dir + "abe_prior.txt", abe_prior.T)
np.savetxt(network_dir + results_dir + "abe_post.txt", abe_post.T)
np.savetxt(network_dir + results_dir + "rmse_prior.txt", rmse_prior.T)
np.savetxt(network_dir + results_dir + "rmse_post.txt", rmse_post.T)
np.savetxt(network_dir + results_dir + "std_prior.txt", std_prior.T)
np.savetxt(network_dir + results_dir + "std_post.txt", std_post.T)

#均值线，reference参考线，RMSE都画

# print(a)
# print(b)
# print(c)