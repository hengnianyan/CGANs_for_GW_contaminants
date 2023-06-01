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


np.random.seed(1000)

os.environ['CUDA_VISIBLE_DEVICES'] = "2" # only one gpu is visible
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


_, realrefers_t, _  = test_set.get_minibatch_well_np(5000)
concen_max = np.max(realrefers_t, axis=0)
concen_max = np.min(realrefers_t, axis=0)






################################################### run model / Monte Carlo
# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)



N_exp = 41  # 实验次数/打井的批次,必须>=2
N_mc = 1    # z向量的个数
N_virtual_obs = 1500 # 某一个井虚拟观测的个数,必须>=batch size
scale_size = 64
min_terval = 4
rows_list = np.arange(1, scale_size-1, min_terval)
# cols_list = np.arange(29, scale_size-1, min_terval)
cols_list = np.arange(26, scale_size-1, min_terval)
N_well_row = rows_list.shape[0]
N_well_col = cols_list.shape[0]


# plot
cmap_well = plt.cm.viridis  # Can be any colormap that you want after the cm   '.
cmap_well.set_bad(color='white')



wd = np.zeros([N_exp, scale_size, scale_size], dtype = 'float32')
optimal_well = np.zeros([N_exp, 2], dtype=int)

rmse = np.zeros([N_exp, 8], dtype = 'float32')
std = np.zeros([N_exp, 8], dtype = 'float32')

_, realrefers_t, _  = training_set.get_minibatch_well_np(N_virtual_obs)  
realrefers_t = realrefers_t[:, 1:2]

##### Begain
for step in range(N_exp):
    print('%03d-th experiment' % (step))


    ##### run the GAN without condition
    print('run the GAN without condition')
    # print(Gs.input_shapes[0][1])
    latents_temp = np.random.randn(N_mc*N_virtual_obs, Gs.input_shapes[0][1]) 
    well_facies_temp = np.zeros([N_mc*N_virtual_obs,2,scale_size,scale_size],dtype='float32')
    labels_temp = np.zeros([N_mc*N_virtual_obs,7],dtype='float32')

    # last steps observation
    if step != 0:
        for step2 in range(step):
            well_facies_temp[:,0,optimal_well[step2,0],optimal_well[step2,1]] = 1
            well_facies_temp[:,1,optimal_well[step2,0],optimal_well[step2,1]] = reals[refer_index,1,optimal_well[step2,0],optimal_well[step2,1]]


    fake_images_out = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
    _,label_est = D2.run(fake_images_out,minibatch_size=400)

    label_est = label_est[:, [0, 1, 2, 5, 6, 3, 4]]

    fake_images_out = (fake_images_out+1)/2

    if step != 0:
        realrefers_t = fake_images_out


    ##### generate data
    well_points = np.zeros([N_virtual_obs*N_well_row*N_well_col, 1, scale_size, scale_size], dtype = int)
    well_facies = np.zeros([N_virtual_obs*N_well_row*N_well_col, 1, scale_size, scale_size],dtype='float32')
    

    
    for r in range(N_virtual_obs):
        
        for j in range(N_well_row):
            for k in range(N_well_col):
                well_points[r*N_well_row*N_well_col+j*N_well_col+k, 0, rows_list[j], cols_list[k]] = 1
 
                well_facies[r*N_well_row*N_well_col+j*N_well_col+k, 0, rows_list[j], cols_list[k]] = realrefers_t[r,0,rows_list[j], cols_list[k]]
                

    
    
    # last steps observation
    if step != 0:
    
        well_points[:,0,optimal_well[0:step,0],optimal_well[0:step,1]] = 1
        well_facies[:,0,optimal_well[0:step,0],optimal_well[0:step,1]] = reals[refer_index,1,optimal_well[0:step,0],optimal_well[0:step,1]]

    well_facies = np.concatenate([well_points, well_facies], 1)


    ##### run the GAN
    print('run the GAN')
    latents_temp = np.random.randn(N_mc*N_virtual_obs*N_well_row*N_well_col, Gs.input_shapes[0][1]) 
    well_facies_temp = np.repeat(well_facies, N_mc, axis=0)
    labels_temp = np.zeros([N_mc*N_virtual_obs*N_well_row*N_well_col,7],dtype='float32')
    

    fake_images_out2 = Gs.run(latents_temp, labels_temp, well_facies_temp,minibatch_size=400)
    _,label_est2 = D2.run(fake_images_out2,minibatch_size=400)

    label_est2 = label_est2[:, [0, 1, 2, 5, 6, 3, 4]]


    fake_images_out2 = (fake_images_out2+1)/2

    ##### calculate the distance; N_mc = 1
    print('calculate the distance')
    for j in range(N_well_row):
        for k in range(N_well_col):
            fake_images_out2_temp = np.zeros_like(fake_images_out) # 临时变量，保证与condition前数据大小一致
            label_est2_temp = np.zeros_like(label_est)
            for r in range(N_virtual_obs):
                fake_images_out2_temp[r,:] = fake_images_out2[r*N_well_row*N_well_col+j*N_well_col+k,:]
                label_est2_temp[r,:] = label_est2[r*N_well_row*N_well_col+j*N_well_col+k,:]
            

            t_wd = samples_entropy(label_est[:,0],label_est2_temp[:,0])
            source_row_wd1 = samples_entropy(label_est[:,1],label_est2_temp[:,1])
            source_col_wd1 = samples_entropy(label_est[:,2],label_est2_temp[:,2])

            source_row_wd2 = samples_entropy(label_est[:,3],label_est2_temp[:,3])
            source_col_wd2 = samples_entropy(label_est[:,4],label_est2_temp[:,4])

            source_row_wd3 = samples_entropy(label_est[:,5],label_est2_temp[:,5])
            source_col_wd3 = samples_entropy(label_est[:,6],label_est2_temp[:,6])


            wd[step,rows_list[j], cols_list[k]] = t_wd + source_row_wd1 + source_col_wd1 + \
                        source_row_wd2 + source_col_wd2 + \
                        source_row_wd3 + source_col_wd3

    ###### experiment/observation
    if step != 0: # 不采相同点
        wd[step,optimal_well[0:step,0],optimal_well[0:step,1]] = 0

    optimal_well[step,0] = np.where(wd[step,:]==np.max(wd[step,:]))[0][0]
    optimal_well[step,1] = np.where(wd[step,:]==np.max(wd[step,:]))[1][0]

    #################################### plot ###############

    

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8, forward=True)
    gci = ax.imshow(wd[step,:], cmap=cmap_well)

    fig.colorbar(gci,ax=ax,format='%.3f')
    plt.savefig(network_dir + results_dir + "w_distance-%03d.png" % step, dpi=200) 


    np.savetxt(network_dir + results_dir + "optimal_well.txt" , optimal_well ,fmt='%3d', delimiter=' ')



    ##### optimal_well plot
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8, forward=True)
    scatter_colors = np.arange(0,N_exp)/(N_exp-1)
    ax.scatter(optimal_well[0:step+1,1]+1,optimal_well[0:step+1,0]+1,s=50,c=scatter_colors[0:step+1])
    ax.axis([1,64,1,64])
    ax.invert_yaxis()

    for i,txt in enumerate(np.arange(0,step+1)):
        # ax.annotate(txt+1,(optimal_well[i,1]-1,optimal_well[i,0]+1))
        ax.text(optimal_well[i,1]+1+1,optimal_well[i,0]+1-1,str(int(txt+1)),fontsize=20)
    plt.savefig(network_dir  + results_dir + "optimal_well-%03d.png" % step, dpi=200) 



    ###### concentration prediction plot, last step
    # Each row has the same input well facies data but different latent vectors
    fig, ax = plt.subplots(1, 6, sharex='col', sharey='row')
    fig.set_size_inches(13, 2, forward=True)
    fig.subplots_adjust(right=0.9,bottom=0.1)

    ax[0].imshow(reals[refer_index,1,:,:], cmap=cmap_well, vmin=0, vmax=1)

    images_plt = fake_images_out[:, 0:1]

    images_plt_average = np.average(images_plt, axis = 0)
    images_plt_variance = np.var(images_plt, axis = 0)

    for j in range(3):
        ax[j+1].imshow(images_plt[j,0,:,:])
    ax[4].imshow(images_plt_average[0], vmin = 0, vmax = 1)   # E-type

    gci = ax[5].imshow(images_plt_variance[0],vmin=0)  # Variance

    cbar_ax = fig.add_axes([0.92,0.1,0.01,0.8]) # 左下宽高
    plt.colorbar(gci,cax=cbar_ax,format='%.3f')

    plt.savefig(network_dir + results_dir + "C simulations conditioned to less well data-%03d.png" % step, dpi=200) 


    ###### Adsorption concentration prediction plot, last step


    ###### label prediction plot, last step
    fig, ax = plt.subplots(1, 7)
    fig.set_size_inches(15, 2, forward=True)


    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink', 'black', 'tab:red','tab:cyan']
    marks = ['-o','-*','-^','-x','-+','-v','-D','-s','-1']
    for i in range(7):
        ax[i].hist(label_est[:,i],bins=50,facecolor=colors[i], edgecolor="black", alpha=0.9)
        ax[i].plot([label_test[refer_index,i],label_test[refer_index,i]] , [0,20],'r-',linewidth=5)
        ax[i].set_xlim(0,1)
        
        
    plt.savefig(network_dir + results_dir + "Label prediction-%03d.png" % step, dpi=200)

    ##### RMSE
    for i in range(7): # label rmse
        rmse[step,i] = np.sqrt(np.mean(np.square(label_est[:,i] - label_test[refer_index,i])))
        std[step,i] = np.sqrt(np.std(np.square(label_est[:,i] - label_test[refer_index,i])))


    rmse[step,7] = np.sqrt(np.mean(np.square(fake_images_out[:, 0:1] - reals[refer_index, 1:2]))) # C rmse
    std[step,7] = np.sqrt(np.std(np.square(fake_images_out[:, 0:1] - reals[refer_index, 1:2])))


    np.savetxt(network_dir + results_dir + "rmse.txt" , rmse, fmt='%.4f', delimiter=' ')
    np.savetxt(network_dir + results_dir + "std.txt" , std, fmt='%.4f', delimiter=' ')

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8, forward=True)
    for i in range(8):
        ax.plot(np.arange(1,step+1+1), rmse[0:step+1,i], marks[i], color=colors[i], linewidth=5, markersize=10)

    ax.set_xlabel("Steps")
    ax.set_ylabel("RMSE")

    plt.savefig(network_dir + results_dir + "RMSE-%03d.png" % step, dpi=200)


    plt.close('all')