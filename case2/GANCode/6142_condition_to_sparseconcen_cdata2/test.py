
import os

import pickle
from turtle import color
import numpy as np
import tensorflow as tf

from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset

np.random.seed(1000)
os.environ['CUDA_VISIBLE_DEVICES']="1" # only one gpu is visible
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

current_dir = os.getcwd()


data_dir_test = '/public/home/yhn/back6_test10/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back6_test10/GANresults/TrainingResults_6142_cdata2_lesswell/007-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'


test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_6132_cdata2_lesswell', max_label_size = 'full')


N_test = 50
N_test2 = 50
reals = np.zeros([N_test2] + test_set.shape, dtype=np.float32)
label_test = np.zeros([N_test, test_set.label_size], dtype=np.float32)

realrefers = np.zeros([N_test] + [2,64,64], dtype=np.float32)

for idx in range(N_test):
    reals_t, label_t = test_set.get_minibatch_imageandlabel_np(1)  
    _, realrefers_t, _  = test_set.get_minibatch_well_np(1)

    reals[idx] = reals_t[0]
    label_test[idx] = label_t[0]

    realrefers[idx] = realrefers_t[0]




################################################### run model


# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)



# Random simulate based on pretrained Network and mannual inspection
well_points = np.zeros([N_test, 1, 64, 64], dtype = int)
well_points2 = np.zeros([N_test, 1, 64, 64], dtype = int)
min_terval = 4
scale_size = 64
well_facies = np.zeros([N_test, 1, 64, 64],dtype='float32')
well_facies2 = np.zeros([N_test, 1, 64, 64],dtype='float32')
for i in range(N_test):
    # well_points_num = np.random.RandomState(3*i).choice(np.arange(10, 50), 1)  # Random choose the expected total number of well points
    # well_points_num = np.random.RandomState(3*i).choice(np.arange(5, 10), 1)  # Random choose the expected total number of well points
    well_points_num = 40
    xs = np.random.RandomState(i*i*i*56).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
    ys = np.random.RandomState(i*i*2+20).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
    # ys = np.random.RandomState(i*i*2+20).choice(np.arange(29, scale_size-1, min_terval), well_points_num)
    well_points[i, 0, xs, ys] = 1
    well_facies[i, 0, xs, ys] = realrefers[i, 1, xs, ys]



well_facies = np.concatenate([well_points, well_facies], 1)

well_points = well_facies[:,0:1]+well_facies[:,1:2]





### Enlarge areas of well points into 2 x 2 only for displaying in following figure
with tf.device('/gpu:0'):
    well_points = tf.cast(well_points, tf.float32)
    wellfacies = tf.nn.max_pool(well_points, ksize = [1,1,2,2], strides=[1,1,1,1], padding='SAME', data_format='NCHW') 


with tf.Session() as sess: 
    sess.run(tf.global_variables_initializer())
    well_points_enlarge = sess.run(wellfacies)



welllocs_el = (well_points_enlarge>0).astype(np.float32)
well_facies_el = (well_points_enlarge - 1)*welllocs_el
well_facies_el = np.concatenate([welllocs_el, well_facies_el], 1) # now wellfacies dimension = [minibatch_in, 2, resolution, resolution]


# make mask of output well facies data only for better displaying in following figure
well_facies_onechannel = well_facies[:,0:1]+well_facies[:,1:2]
well_facies_onechannel_mask = np.ma.masked_where(well_facies_onechannel == 0, well_facies_onechannel-1)
well_facies_el_onechannel = well_facies_el[:,0:1]+well_facies_el[:,1:2]
well_facies_el_onechannel_mask = np.ma.masked_where(well_facies_el_onechannel == 0, well_facies_el_onechannel-1)


cmap_well = plt.cm.viridis  # Can be any colormap that you want after the cm   '.
# cmap_well = plt.cm.jet
cmap_well.set_bad(color='white')


########### concentration
vmax = 0.7
# Each row has the same input well facies data but different latent vectors
fig, ax = plt.subplots(8, 9, sharex='col', sharey='row')
fig.set_size_inches(15, 10.5, forward=True)

for i in range (8):
    ax[i, 0].imshow(well_facies_el_onechannel_mask[i,0], cmap=cmap_well, vmin=0, vmax=vmax)

    ax[i, 1].imshow(realrefers[i,1,:,:], cmap=cmap_well, vmin=0, vmax=vmax)


    latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

    labels_plt = np.repeat(label_test[i:i+1], 500, axis=0)
    well_facies_plt = np.repeat(well_facies[i:i+1], 500, axis=0)


    # Run the generator to produce a set of images.
    fake_plt = G.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
    images_plt = (fake_plt[:,0:1]+1)/2


    images_plt_average = np.average(images_plt, axis = 0)
    images_plt_variance = np.var(images_plt, axis = 0)
    
    for j in range(5):
        ax[i, j+2].imshow(images_plt[j,0,:,:])
    ax[i, 7].imshow(images_plt_average[0], vmin = 0, vmax = vmax)   # E-type

    gci = ax[i, 8].imshow(images_plt_variance[0],vmin=0)  # Variance
    fig.colorbar(gci,ax=ax[i, 8],format='%.3f')

plt.savefig(network_dir + "C simulations conditioned to less well data.png", dpi=200) 



# time label
_,time_est = D2.run(reals[0:N_test2, 1:2]*2-1,minibatch_size=16)

fig, ax = plt.subplots(1, 3)
fig.set_size_inches(16, 5, forward=True)
colors = ['tab:blue', 'tab:green', 'black']
for j in range(3):
    if j == 0:
        for i in range(3):
            ax[i].scatter(label_test[0:N_test2,i], time_est[0:N_test2,i], c=colors[j])
            # calc the trendline
            z = np.polyfit(label_test[0:N_test2,i], time_est[0:N_test2,i], 1)
            p = np.poly1d(z)
            ax[i].plot(label_test[0:N_test2,i],label_test[0:N_test2,i],"r-")

            min_plt = 0
            max_plt = 1

            ax[i].plot(np.array([min_plt,max_plt]),p(np.array([min_plt,max_plt])),'--', color=colors[j])

            # the line equation:
            ax[i].text(min_plt,max_plt*3/4 + j*0.1,"y=%.3fx+(%.3f)"%(z[0],z[1]))
            print ("y=%.6fx+(%.6f)"%(z[0],z[1]))

            ax[i].set_xlabel("Reference labels")
            ax[i].set_ylabel("Estimated labels")
    else:
        for i in range(1,3):
            ax[i].scatter(label_test[0:N_test2, j*2+i], time_est[0:N_test2, j*2+i], c=colors[j])
            # calc the trendline
            z = np.polyfit(label_test[0:N_test2, j*2+i], time_est[0:N_test2, j*2+i], 1)
            p = np.poly1d(z)
            ax[i].plot(label_test[0:N_test2, j*2+i],label_test[0:N_test2, j*2+i],"r-")


            ax[i].plot(np.array([min_plt,max_plt]),p(np.array([min_plt,max_plt])),'--', color=colors[j])

            # the line equation:
            ax[i].text(min_plt,max_plt*3/4 + j*0.1,"y=%.3fx+(%.3f)"%(z[0],z[1]))
            print ("y=%.6fx+(%.6f)"%(z[0],z[1]))

            ax[i].set_xlabel("Reference labels")
            ax[i].set_ylabel("Estimated labels")
plt.savefig(network_dir +"Labels fake vs real.png", dpi=200)
