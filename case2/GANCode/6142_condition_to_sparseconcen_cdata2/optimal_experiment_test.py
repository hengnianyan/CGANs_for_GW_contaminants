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

from sklearn.gaussian_process import GaussianProcessRegressor
 
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# from scipy.stats import wasserstein_distance

np.random.seed(1000)
# os.environ.update(config.env)
os.environ['CUDA_VISIBLE_DEVICES']="2" # only one gpu is visible
tfutil.init_tf(config.tf_config)

current_dir = os.getcwd()



data_dir_test = '/public/home/yhn/back6_test10/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back6_test10/GANresults/TrainingResults_6142_cdata2_lesswell/007-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'
results_dir = 'optimal_exp2_results_refer6_Nexp40_Cprior_wdlabel_N_vobs2000_reverseKL_rematch/'

refer_index = 6 # 第几个样本为参考值

test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_6132_cdata2_lesswell', max_label_size = 'full')


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

# N_repeat = 1
N_repeat = 10 # 重复随机选择10组观测井组合
N_exp = 41  # 实验次数/打井的批次,必须>=2
N_mc = 1    # z向量的个数
N_virtual_obs = 1000 # 某一个井虚拟观测的个数,必须>=batch size
scale_size = 64
min_terval = 4
rows_list = np.arange(1, scale_size-1, min_terval)
cols_list = np.arange(26, scale_size-1, min_terval)
N_well_row = rows_list.shape[0]
N_well_col = cols_list.shape[0]



rmse_optimal = np.loadtxt(network_dir + results_dir + "rmse.txt")
std_optimal = np.loadtxt(network_dir + results_dir + "std.txt")
rmse_random = np.zeros([N_repeat,N_exp,8],dtype='float32') # [重复，试验次数，rmse种类]


########### 均匀观测下的克里金插值
# Build a model
kernel = 1.0 * RBF(1.0)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
 
# Some data
grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63

rows = np.arange(1, scale_size-1, 8)
cols = np.arange(26, scale_size-1, 8)
row_cols = np.meshgrid(rows,cols)
rows = row_cols[0].flatten()
cols = row_cols[1].flatten()


base_points = reals[refer_index,1,rows,cols].reshape((-1, 1))
grid_X_flat = grid_X[rows,cols].reshape((-1, 1))
grid_Y_flat = grid_Y[rows,cols].reshape((-1, 1))

xobs = np.concatenate((grid_X_flat[:, 0:1], grid_Y_flat[:, 0:1]), axis = 1)
yobs = base_points[:, 0:1]
 
# Fit the model to the data (optimize hyper parameters)
gp.fit(xobs, yobs)
 
# Plot points and predictions
grid_X_flat = grid_X.reshape((-1, 1))
grid_Y_flat = grid_Y.reshape((-1, 1))
x_test = np.concatenate((grid_X_flat[:, 0:1], grid_Y_flat[:, 0:1]), axis = 1)

means, sigmas = gp.predict(x_test, return_std=True)
means = means.reshape((-1, scale_size, scale_size))

cmap_well = plt.cm.viridis  # Can be any colormap that you want after the cm   '.
cmap_well.set_bad(color='white')

fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
fig.set_size_inches(8, 8, forward=True)
fig.subplots_adjust(right=0.9,bottom=0.1)

ax.imshow(means[0], cmap=cmap_well, vmin=0, vmax=1)


plt.savefig(network_dir + results_dir + "kriging.png" , dpi=200) 


############# 均匀观测下的GAN插值
###### concentration prediction plot, last step
# Each row has the same input well facies data but different latent vectors
fig, ax = plt.subplots(1, 6, sharex='col', sharey='row')
fig.set_size_inches(13, 2, forward=True)
fig.subplots_adjust(right=0.9,bottom=0.1)

ax[0].imshow(reals[refer_index,1,:,:], cmap=cmap_well, vmin=0, vmax=1)


latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1]) 
well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')
well_facies_temp[:, 0, rows, cols] = 1
well_facies_temp[:, 1, rows, cols] = reals[refer_index, 1, rows, cols]


# run the gan
fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
images_plt = (fake_images_out+1)/2

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

for j in range(3):
    ax[j+1].imshow(images_plt[j,0,:,:])
ax[4].imshow(images_plt_average[0], vmin = 0, vmax = 1)   # E-type

gci = ax[5].imshow(images_plt_variance[0],vmin=0)  # Variance

cbar_ax = fig.add_axes([0.92,0.1,0.01,0.8]) # 左下宽高
plt.colorbar(gci,cax=cbar_ax,format='%.3f')

plt.savefig(network_dir + results_dir + "GAN_uniform_refer.png", dpi=200) 




#########  random experiment design


for repeat in range(N_repeat):
    print('###########%03d-th repeat#######' % (repeat))
    for step in range(N_exp):
        print('%03d-th experiment' % (step))

        latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1]) 
        well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
        labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')


        rows = np.random.choice(np.arange(1, scale_size-1, min_terval), step)
        # cols = np.random.choice(np.arange(29, scale_size-1, min_terval), step)
        cols = np.random.choice(np.arange(26, scale_size-1, min_terval), step)
        well_facies_temp[:,0,rows,cols] = 1
        well_facies_temp[:,1,rows,cols] = reals[refer_index,1,rows,cols]
            

        fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
        _,label_est = D2.run(fake_images_out,minibatch_size=400)
        fake_images_out = (fake_images_out[:,0:1]+1)/2


        ##### RMSE
        for i in range(7): # label rmse
            rmse_random[repeat,step,i] = np.sqrt(np.mean(np.square(label_est[:,i] - label_test[refer_index,i])))

        rmse_random[repeat,step,7] = np.sqrt(np.mean(np.square(fake_images_out - reals[refer_index,1:2]))) # C rmse



np.save(network_dir + results_dir + "rmse_random.npy" , rmse_random)







########################### 均匀观测

latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1]) 
well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')


rows = np.arange(1, scale_size-1, 4) 
cols = np.arange(26, scale_size-1, 4)
row_cols = np.meshgrid(rows,cols)
rows = row_cols[0].flatten()
cols = row_cols[1].flatten()
well_facies_temp[:,0,rows,cols] = 1
well_facies_temp[:,1,rows,cols] = reals[refer_index,1,rows,cols]
    

fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
_,label_est = D2.run(fake_images_out,minibatch_size=400)
fake_images_out = (fake_images_out[:,0:1]+1)/2

rmse_uniform = np.zeros([1,8],dtype='float32')
##### RMSE
for i in range(7): # label rmse
    rmse_uniform[0,i] = np.sqrt(np.mean(np.square(label_est[:,i] - label_test[refer_index,i])))

rmse_uniform[0,7] = np.sqrt(np.mean(np.square(fake_images_out - reals[refer_index, 1:2]))) # C rmse




################################### plot
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'black', 'tab:red','tab:cyan']
marks = ['-o','-*','-^','-x','-+','-v','-D','-s','-1']
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8, forward=True)
xt = np.tile(np.arange(1,N_exp+1),(N_repeat,1))
for i in range(8):
    ax.plot(np.arange(1,N_exp+1), rmse_optimal[0:N_exp,i], marks[i], color=colors[i], linewidth=5, markersize=10)
    ax.plot([1,41],[rmse_uniform[0,i],rmse_uniform[0,i]], '--', color=colors[i], linewidth=2.5)

    ax.scatter(xt,rmse_random[:,:,i],s=10,c=colors)

ax.set_xlabel("Steps")
ax.set_ylabel("RMSE")
# ax.set_yscale('log') 


plt.savefig(network_dir + results_dir + "RMSE_test.png" , dpi=200)


####################  多类RMSE求平均 
fig, ax = plt.subplots(1, 1)
fig.set_size_inches(8, 8, forward=True)
xt = np.tile(np.arange(1,N_exp+1),(N_repeat,1))

ax.plot(np.arange(1,N_exp+1), np.mean(rmse_optimal[0:N_exp,:],axis=1), '-o', color='tab:blue', linewidth=5, markersize=10)
ax.plot([1,41],[np.mean(rmse_uniform),np.mean(rmse_uniform)], '--', color='tab:red', linewidth=2.5)

ax.scatter(xt,np.mean(rmse_random,axis=2),s=10,c='black')

ax.set_xlabel("Steps")
ax.set_ylabel("RMSE")
# ax.set_yscale('log') 


plt.savefig(network_dir + results_dir + "RMSE_mean_test.png" , dpi=200)




#################### 逐步增加均匀观测
aaa = np.arange(16,7,-1).shape[0]
rmse_uniform_rise = np.zeros([aaa,8],dtype='float32')
well_num_rise = np.zeros([aaa,1],dtype='float32') # 记录观测数目

# [16, 15, ..., 8]
for step,txt in enumerate(np.arange(16,7,-1)):


    print('%03d-th experiment' % (step))

    latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1]) 
    well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
    labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')


    rows = np.arange(1, scale_size-1, txt) 
    cols = np.arange(26, scale_size-1, txt)
    row_cols = np.meshgrid(rows,cols)
    rows = row_cols[0].flatten()
    cols = row_cols[1].flatten()
    well_num_rise[step,0] = rows.shape[0]

    well_facies_temp[:,0,rows,cols] = 1
    well_facies_temp[:,1,rows,cols] = reals[refer_index,1,rows,cols]

        

    fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
    _,label_est = D2.run(fake_images_out,minibatch_size=400)
    fake_images_out = (fake_images_out[:,0:1]+1)/2


    ##### RMSE
    for i in range(7): # label rmse
        rmse_uniform_rise[step,i] = np.sqrt(np.mean(np.square(label_est[:,i] - label_test[refer_index,i])))

    rmse_uniform_rise[step,7] = np.sqrt(np.mean(np.square(fake_images_out - reals[refer_index,1:2]))) # C rmse




marks = ['o','*','^','x']

plt.rc('font',family='Times New Roman') #全局字体
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
fig.set_size_inches(6.5, 6, forward=True)
fig.dpi = 600
ftsize = 14


s1, = ax.plot(np.arange(1,N_exp+1), np.mean(rmse_optimal[0:N_exp,:],axis=1), '-o', color='tab:blue', linewidth=5, markersize=10)


s2 = ax.scatter(xt,np.mean(rmse_random,axis=2),s=10,c='black')
s3 = ax.scatter(well_num_rise[:,0],np.mean(rmse_uniform_rise,axis=1), s=100, c='tab:red', marker='*')



ax.set_xlabel("Number of observation locations", fontsize = ftsize)
ax.set_ylabel("RMSE", fontsize = ftsize)

plt.xticks(fontsize=ftsize)
plt.yticks(fontsize=ftsize)
# ax.set_yscale('log') 


#添加图例
ax.legend(handles=[s1, s2, s3], \
        labels=['Optimized', 'Random', 'Uniform'], markerscale=1, fontsize=ftsize)


plt.savefig(network_dir + results_dir + "RMSE_uniform_rise_test.png" , dpi=600)


np.savetxt(network_dir + results_dir + "rmse_uniform_rise.txt" , rmse_uniform_rise, fmt='%.4f', delimiter=' ')
np.savetxt(network_dir + results_dir + "well_num_rise.txt" , well_num_rise ,fmt='%3d', delimiter=' ')
