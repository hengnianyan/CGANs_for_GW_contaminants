
import os
# import sys
import pickle
import numpy as np
import tensorflow as tf
# import PIL.Image
import matplotlib as mpl
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset
from pykrige.ok import OrdinaryKriging



os.environ['CUDA_VISIBLE_DEVICES']="3" # only one gpu is visible

np.random.seed(100)

tfutil.init_tf(config.tf_config)




def distance_matrix(x0, y0, x1, y1):
    """ Make a distance matrix between pairwise observations.
    Note: from  
    """
    
    obs = np.vstack((x0, y0)).T
    interp = np.vstack((x1, y1)).T

    d0 = np.subtract.outer(obs[:,0], interp[:,0])
    d1 = np.subtract.outer(obs[:,1], interp[:,1])
    
    # calculate hypotenuse
    return np.hypot(d0, d1)


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


data_dir_test = '/public/home/yhn/back3_test7/tfdataset/'


# trained generator directory path; please replace it with your own path.
# network_dir = '/public/home/yhn/back3_test4.1/GANresults/TrainingResults_212_lesswell/002-pgan-2gpu-CondWellrealreferpenalty/'
network_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell/000-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'


test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TrainingData_3342_chdata3_lesswell', max_label_size = 'full')


N_test = 500
N_test2 = 500
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



# Random simulate based on pretrained Network and mannual inspection
well_points = np.zeros([N_test, 1, 64, 64], dtype = int)
well_points_num = 40
base_point_xs = np.zeros([N_test, well_points_num], dtype = int)
base_point_ys = np.zeros([N_test, well_points_num], dtype = int)
min_terval = 4
scale_size = 64
well_facies = np.zeros([N_test, 1, 64, 64],dtype='float32')


well_facies_none = np.concatenate([well_points, well_facies], 1)

for i in range(N_test):

    xs = np.random.RandomState(i*i*2+56).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
    ys = np.random.RandomState(i*i*2+20).choice(np.arange(1, scale_size-1, min_terval), well_points_num)

    well_points[i, 0, xs, ys] = 1
    well_facies[i, 0, xs, ys] = realrefers[i, 0, xs, ys]



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
# cmap_well = plt.cm.viridis  # Can be any colormap that you want after the cm   '.
# cmap_well = plt.cm.jet
cmap_well = plt.cm.GnBu
cmap_well.set_bad(color='white')
vmin = 0 * 50
vmax = 0.7 * 50
levels = 10



latents_plt = np.random.randn(N_test, Gs.input_shapes[0][1]) 

labels_plt = label_test
well_facies_plt = well_facies
well_facies_none_plt = well_facies_none


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

fake_plt = Gs.run(latents_plt, labels_plt, well_facies_none_plt,minibatch_size=16) #, probimages_plt
images_none_plt = (fake_plt[:,0:1]+1)/2

########### concentration
# Each row has the same input well facies data but different latent vectors
plt.rc('font',family='Times New Roman') #全局字体
fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
# fig, ax = plt.subplots(1, 2)
# fig, ax = plt.subplots(2, 4)
fig.set_size_inches(5, 4, forward=True)
fig.dpi = 600
ftsize = 12
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(0.2 * 50) # 每隔整数取tick


grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63

abc_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


colors = ['tab:blue', 'tab:orange', 'tab:green', 'black']

# 做lnC的分布图

# 重新取refer样本
realrefers = np.zeros([N_test] + [1,64,64], dtype=np.float32)

for idx in range(N_test):
    reals_t, label_t = test_set.get_minibatch_imageandlabel_np(1)  
    _, realrefers_t  = test_set.get_minibatch_well_np(1)
    # _, concen, lastconcensrefers_t, realrefers_t,velocitys_t,labels_shuffle_t  = test_set.get_minibatch_well_np(1)
    reals[idx] = reals_t[0]
    label_test[idx] = label_t[0]
    realrefers[idx] = realrefers_t[0]



realrefers[realrefers<0] = 0
images_plt[images_plt<0] = 0
images_none_plt[images_none_plt<0] = 0



[_, _, bar1] = ax.hist(realrefers.reshape((-1, 1)), range=(0,1), bins=50, \
        density=True, facecolor=plt.cm.Set2(0), alpha=0.7)
[_, _, bar2] = ax.hist(images_plt.reshape((-1, 1)), range=(0,1), bins=50, \
        density=True, facecolor=plt.cm.Set2(1), alpha=0.7)

ax.set_yscale("log", basey=10) # y 轴上以10为底数呈对数显示，2、3表示会标记出2倍、3倍的位置
ax.set_xlim(0,1)
ax.grid(axis='y', ls='--', alpha=0.8)
ax.set_xlabel("Concentration", fontsize=ftsize)



ax.legend(handles=[bar1[0], bar2[0]], \
        labels=['Data','CGANs'], markerscale=1.5, fontsize=ftsize)



plt.tight_layout()
plt.savefig(network_dir + "simulations conditioned to less well data_prior_dis2.png", dpi=600) 
