
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


N_test = 30
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


well_points_num_list = [10, 20, 30, 40, 50]
RMSE_ok = np.zeros([N_test, 5], dtype=np.float32)
RMSE_idw = np.zeros([N_test, 5], dtype=np.float32)
RMSE_cgans = np.zeros([N_test, 5], dtype=np.float32)

min_terval = 4
scale_size = 64
grid_x = np.arange(0, scale_size, 1, dtype='float32')
grid_y = np.arange(0, scale_size, 1, dtype='float32')
grid_X, grid_Y = np.meshgrid(grid_x, grid_y) # x轴从左到右，从0-63，y轴从上到下，从0-63


for i in range(5):
    well_points_num = well_points_num_list[i]
    well_points = np.zeros([N_test, 1, 64, 64], dtype = int)
    base_point_xs = np.zeros([N_test, well_points_num], dtype = int)
    base_point_ys = np.zeros([N_test, well_points_num], dtype = int)

    well_facies = np.zeros([N_test, 1, 64, 64],dtype='float32')

    for j in range(N_test):
        xs = np.random.RandomState(j*j*j*56).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
        ys = np.random.RandomState(j*j*2+20).choice(np.arange(1, scale_size-1, min_terval), well_points_num)
        base_point_xs[j, :] = xs
        base_point_ys[j, :] = ys
        
        well_points[j, 0, xs, ys] = 1
        well_facies[j, 0, xs, ys] = realrefers[j, 0, xs, ys]



    well_facies = np.concatenate([well_points, well_facies], 1)
    well_points = well_facies[:,0:1]+well_facies[:,1:2]

    for j in range(N_test):
        latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 
       
        labels_plt = np.repeat(label_test[j:j+1], 500, axis=0)
        well_facies_plt = np.repeat(well_facies[j:j+1], 500, axis=0)
      

        # Run the generator to produce a set of images.
        fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
        images_plt = (fake_plt[:,0:1]+1)/2

        images_plt[images_plt<0] = 0

        images_plt_average = np.average(images_plt, axis = 0)

        RMSE_cgans[j, i] = np.sqrt((((images_plt_average[0] - realrefers[j,0,:,:]) * 50) ** 2).mean())
        
        print(j)
        ############### 克里金插值
        base_points = realrefers[j, 0, base_point_xs[j, :], base_point_ys[j, :]].reshape((-1, 1))
        grid_X_flat = grid_X[base_point_xs[j, :], base_point_ys[j, :]].reshape((-1, 1))
        grid_Y_flat = grid_Y[base_point_xs[j, :], base_point_ys[j, :]].reshape((-1, 1))
        ok3d = OrdinaryKriging(grid_X_flat[:, 0], grid_Y_flat[:, 0], base_points[:, 0], variogram_model="gaussian", nlags=10) # 模型
        # variogram_model是变差函数模型，pykrige提供 linear, power, gaussian, spherical, exponential, hole-effect几种variogram_model可供选择，默认的为linear模型。

        k3d1, ss3d = ok3d.execute("grid", grid_x, grid_y) # k3d1是结果，给出了每个网格点处对应的值

        RMSE_ok[j, i] = np.sqrt((((k3d1 - realrefers[j,0,:,:]) * 50) ** 2).mean())



        ###############  IDW 插值
        idw1 = simple_idw(grid_X_flat[:, 0], grid_Y_flat[:, 0], base_points[:, 0],\
                    grid_X.reshape((-1, 1)), grid_Y.reshape((-1, 1)), power=2)
        idw1 = idw1.reshape((scale_size, scale_size))

        RMSE_idw[j, i] = np.sqrt((((idw1 - realrefers[j,0,:,:]) * 50) ** 2).mean())





##############画图
plt.rc('font',family='Times New Roman') #全局字体


fig, ax = plt.subplots(1, 1, sharex='col', sharey='row')
fig.set_size_inches(8, 4, forward=True)
fig.dpi = 600
ftsize = 12
formatter = mpl.ticker.StrMethodFormatter('{x:.1f}')
locator = mpl.ticker.MultipleLocator(0.2 * 50) # 每隔整数取tick

x1 = np.arange(1,20,4)
x2 = x1+1
x3 = x1+2

box1 = ax.boxplot(RMSE_ok, positions=x1, patch_artist=True, showmeans=False, showfliers=False,
            boxprops={"facecolor": plt.cm.Set2(0),
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})
box2 = ax.boxplot(RMSE_idw, positions=x2, patch_artist=True, showmeans=False, showfliers=False,
            boxprops={"facecolor": plt.cm.Set2(1),
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})
box3 = ax.boxplot(RMSE_cgans, positions=x3, patch_artist=True, showmeans=False, showfliers=False,
            boxprops={"facecolor": plt.cm.Set2(2),
                      "edgecolor": "grey",
                      "linewidth": 0.5},
            medianprops={"color": "k", "linewidth": 0.5},
            meanprops={'marker':'+',
                       'markerfacecolor':'k',
                       'markeredgecolor':'k',
                       'markersize':5})


np.savetxt(network_dir + "ok_rmse.txt", RMSE_ok)
np.savetxt(network_dir + "idw_rmse.txt", RMSE_idw)
np.savetxt(network_dir + "cgans_rmse.txt", RMSE_cgans)


ax.set_xlim(0, 20)
ax.set_ylim(0, 7)

ax.set_xticks([2, 6, 10, 14, 18])
ax.set_xticklabels(well_points_num_list, fontsize=ftsize)
ax.set_yticks(np.arange(0, 8))
ax.set_yticklabels(np.arange(0, 8).tolist(), fontsize=ftsize)

ax.set_ylabel('RMSE', fontsize=ftsize)
ax.set_xlabel('Number of observations', fontsize=ftsize)
ax.grid(axis='y', ls='--', alpha=0.8)

# 给箱体添加图例，每类箱线图中取第一个颜色块用于代表图例
ax.legend(handles=[box1['boxes'][0], box2['boxes'][0], box3['boxes'][0]], \
        labels=['OK', 'IDW', 'CGANs'], markerscale=1.5, fontsize=ftsize)




plt.savefig(network_dir + "simulations conditioned to less well data_boxrmse.png", dpi=600) 



