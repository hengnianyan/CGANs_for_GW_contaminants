
import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib as mpl
from matplotlib import use
use('Agg')
import matplotlib.pyplot as plt
import tfutil
import config
import dataset
from pykrige.ok import OrdinaryKriging

np.random.seed(1000)

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

def simple_idw(x, y, z, xi, yi, power=1):
    """ Simple inverse distance weighted (IDW) interpolation 
    Weights are proportional to the inverse of the distance, so as the distance
    increases, the weights decrease rapidly.
    The rate at which the weights decrease is dependent on the value of power.
    As power increases, the weights for distant points decrease rapidly.
    """
    num_grids = xi.size
    idw1 = np.zeros([1, num_grids])

    for i in range(num_grids):
        dist = distance_matrix(x, y, xi[i], yi[i])

        # In IDW, weights are 1 / distance
        weights = 1.0/(dist+1e-12)**power

        # Make weights sum to one
        weights /= weights.sum(axis=0)

        # Multiply the weights for each interpolated point by all observed Z-values
        idw1[0, i] = np.dot(weights.T, z)

    return idw1

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
os.environ['CUDA_VISIBLE_DEVICES']="3" # only one gpu is visible


data_dir_test = '/public/home/yhn/back3_test7/tfdataset/'


# trained generator directory path; please replace it with your own path.
network_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell/000-pgan-2gpu-CondWell/'
network_name = 'network-snapshot-004500.pkl'

test_set = dataset.load_dataset(data_dir=data_dir_test, verbose=True, tfrecord_dir='TestData_3342_chdata3_lesswell', max_label_size = 'full')


N_test = 50
N_test2 = 50
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
for i in range(N_test):

    xs = np.random.RandomState(i*i*i*56).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
    ys = np.random.RandomState(i*i*2+20).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
    base_point_xs[i, :] = xs
    base_point_ys[i, :] = ys

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


# 只展示两个reference
realrefers = realrefers[[0, 5],:,:,:]
label_test = label_test[[0, 5]]
well_facies = well_facies[[0, 5]]
well_facies_el = well_facies_el[[0, 5]]


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

########### concentration
# Each row has the same input well facies data but different latent vectors
plt.rc('font',family='Times New Roman') #全局字体
plt.rc('mathtext',fontset='stix') # 公式字体
fig, ax = plt.subplots(2, 4, sharex='col', sharey='row')
# fig, ax = plt.subplots(2, 4)
fig.set_size_inches(10, 5, forward=True)
fig.dpi = 600
ftsize = 12
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(0.2 * 50) # 每隔整数取tick


grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63

abc_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

for i in range (2):
    ############## 参考图
    base_points = realrefers[i, 0, base_point_xs[i, :], base_point_ys[i, :]].reshape((-1, 1))
    grid_X_flat = grid_X[base_point_xs[i, :], base_point_ys[i, :]].reshape((-1, 1))
    grid_Y_flat = grid_Y[base_point_xs[i, :], base_point_ys[i, :]].reshape((-1, 1))


    ax[i, 0].contourf(grid_X, grid_Y, realrefers[i,0,:,:] * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
    ax[i, 0].contour(grid_X, grid_Y, realrefers[i,0,:,:] * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
    ax[i, 0].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')

    
    # xy坐标轴设置
    ax[i, 0].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
    ax[i, 0].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)
    # ax[i, 0].set_yticks(np.linspace(63, 3, 5))
    ax[i, 0].set_yticks(np.linspace(0, 60, 5))
    ax[i, 0].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)
    
    ax[i, 0].set_title('('+ abc_list[i*4 + 0] + ')' + ' Reference')

    

    latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 
    labels_plt = np.repeat(label_test[i:i+1], 500, axis=0)
    well_facies_plt = np.repeat(well_facies[i:i+1], 500, axis=0)
    

    # Run the generator to produce a set of images.
    fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
    images_plt = (fake_plt[:,0:1]+1)/2

    images_plt[images_plt<0] = 0

    images_plt_average = np.average(images_plt, axis = 0)
    images_plt_variance = np.var(images_plt, axis = 0)
    
    
    ############### 克里金插值
    ok3d = OrdinaryKriging(grid_X_flat[:, 0], grid_Y_flat[:, 0], base_points[:, 0], variogram_model="gaussian", nlags=20) # 模型
    # variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。

    k3d1, ss3d = ok3d.execute("grid", grid_x, grid_y) # k3d1是结果，给出了每个网格点处对应的值



    ax[i, 1].contourf(grid_X, grid_Y, k3d1 * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
    ax[i, 1].contour(grid_X, grid_Y, k3d1 * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
    ax[i, 1].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')
    
    rmse = np.sqrt((((k3d1 - realrefers[i,0,:,:]) * 50) ** 2).mean())
    ax[i, 1].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
                bbox={'facecolor':'white', 'edgecolor':'white'})

    

    # xy坐标轴设置
    ax[i, 1].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
    ax[i, 1].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


    ax[i, 1].set_title('('+ abc_list[i*4 + 1] + ')' + ' OK')

    ###############  IDW 插值
    idw1 = simple_idw(grid_X_flat[:, 0], grid_Y_flat[:, 0], base_points[:, 0],\
                    grid_X.reshape((-1, 1)), grid_Y.reshape((-1, 1)), power=2)


    idw1 = idw1.reshape((scale_size, scale_size))


    ax[i, 2].contourf(grid_X, grid_Y, idw1 * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
    ax[i, 2].contour(grid_X, grid_Y, idw1 * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
    ax[i, 2].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')
    
    rmse = np.sqrt((((idw1 - realrefers[i,0,:,:]) * 50) ** 2).mean())
    ax[i, 2].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
                bbox={'facecolor':'white', 'edgecolor':'white'})
    
    # xy坐标轴设置
    ax[i, 2].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
    ax[i, 2].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


    ax[i, 2].set_title('('+ abc_list[i*4 + 2] + ')' + ' IDW')

    ############## GAN插值
    im = ax[i, 3].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
    ax[i, 3].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
                levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
    ax[i, 3].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')

    
    rmse = np.sqrt((((images_plt_average[0] - realrefers[i,0,:,:]) * 50) ** 2).mean())
    ax[i, 3].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
                bbox={'facecolor':'white', 'edgecolor':'white'})

    
    # xy坐标轴设置
    ax[i, 3].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
    ax[i, 3].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


    ax[i, 3].set_title('('+ abc_list[i*4 + 3] + ')' + ' CGANs')

    # colorbar设置
    cax = add_right_cax(ax[i, 3], pad=0.02, width=0.02)
    cbar = fig.colorbar(im, cax=cax, ticks=locator, format=formatter)
    cbar.ax.tick_params(labelsize=ftsize)





plt.savefig(network_dir + "simulations conditioned to less well data_test2.png", dpi=600) 


