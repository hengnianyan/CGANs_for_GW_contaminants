
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
network_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell/000-pgan-2gpu-CondWell/'


########################## load data
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
    # ys = np.random.RandomState(i*i*2+20).choice(np.arange(29, scale_size-1, min_terval), well_points_num)
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



#################### Figure settings
cmap_well = plt.cm.GnBu
cmap_well.set_bad(color='white')
vmin = 0 * 50
vmax = 0.7 * 50
levels = 10

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


################################################### Reference
refer_index = 0

base_points = realrefers[refer_index, 0, base_point_xs[refer_index, :], base_point_ys[refer_index, :]].reshape((-1, 1))
grid_X_flat = grid_X[base_point_xs[refer_index, :], base_point_ys[refer_index, :]].reshape((-1, 1))
grid_Y_flat = grid_Y[base_point_xs[refer_index, :], base_point_ys[refer_index, :]].reshape((-1, 1))


ax[0, 0].contourf(grid_X, grid_Y, realrefers[refer_index,0,:,:] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[0, 0].contour(grid_X, grid_Y, realrefers[refer_index,0,:,:] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[0, 0].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


# xy坐标轴设置
ax[0, 0].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[0, 0].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[0, 0].set_yticks(np.linspace(0, 60, 5))
ax[0, 0].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[0, 0].set_title('('+ abc_list[0] + ')' + ' Reference')

################################################ Subplot 1
network_name = 'network-snapshot-000160.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[0, 1].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[0, 1].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[0, 1].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[0, 1].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

# xy坐标轴设置
ax[0, 1].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[0, 1].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[0, 1].set_yticks(np.linspace(0, 60, 5))
ax[0, 1].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


ax[0, 1].set_title('('+ abc_list[1] + ')' + '160 (' + '$10^3$' + ') images')



################################################ Subplot 2
network_name = 'network-snapshot-000482.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[0, 2].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[0, 2].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[0, 2].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[0, 2].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

# xy坐标轴设置
ax[0, 2].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[0, 2].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[0, 2].set_yticks(np.linspace(0, 60, 5))
ax[0, 2].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)



ax[0, 2].set_title('('+ abc_list[2] + ')' + '482 (' + '$10^3$' + ') images')


################################################ Subplot 3
network_name = 'network-snapshot-000802.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[0, 3].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[0, 3].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[0, 3].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[0, 3].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

# xy坐标轴设置
ax[0, 3].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[0, 3].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[0, 3].set_yticks(np.linspace(0, 60, 5))
ax[0, 3].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


ax[0, 3].set_title('('+ abc_list[3] + ')' + '802 (' + '$10^3$' + ') images')


################################################ Subplot 4
network_name = 'network-snapshot-001443.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[1, 0].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[1, 0].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[1, 0].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[1, 0].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})


# xy坐标轴设置
ax[1, 0].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[1, 0].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[1, 0].set_yticks(np.linspace(0, 60, 5))
ax[1, 0].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


ax[1, 0].set_title('('+ abc_list[4] + ')' + '1443 (' + '$10^3$' + ') images')


################################################ Subplot 5
network_name = 'network-snapshot-002084.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[1, 1].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[1, 1].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[1, 1].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[1, 1].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

# xy坐标轴设置
ax[1, 1].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[1, 1].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[1, 1].set_yticks(np.linspace(0, 60, 5))
ax[1, 1].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)



ax[1, 1].set_title('('+ abc_list[5] + ')' + '2084 (' + '$10^3$' + ') images')


################################################ Subplot 6
network_name = 'network-snapshot-003366.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[1, 2].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[1, 2].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[1, 2].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[1, 2].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

ax[1, 2].set_xticks(np.linspace(0, 60, 5)) # 将行列号[0,1,2,...63]显示为距离m
ax[1, 2].set_xticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)

ax[1, 2].set_yticks(np.linspace(0, 60, 5))
ax[1, 2].set_yticklabels(np.round(np.linspace(0, 93.75, 5), 0).astype(int).tolist(), fontsize = ftsize)


ax[1, 2].set_title('('+ abc_list[6] + ')' + '3366 (' + '$10^3$' + ') images')



################################################ Subplot 7
network_name = 'network-snapshot-004500.pkl'
# index = 1

# Import networks.
with open(network_dir+network_name, 'rb') as file:
    G, D, D2, Gs = pickle.load(file)

latents_plt = np.random.randn(500, Gs.input_shapes[0][1]) 

labels_plt = np.repeat(label_test[refer_index:refer_index+1], 500, axis=0)
well_facies_plt = np.repeat(well_facies[refer_index:refer_index+1], 500, axis=0)


# Run the generator to produce a set of images.
fake_plt = Gs.run(latents_plt, labels_plt, well_facies_plt,minibatch_size=16) #, probimages_plt
images_plt = (fake_plt[:,0:1]+1)/2

images_plt[images_plt<0] = 0

images_plt_average = np.average(images_plt, axis = 0)
images_plt_variance = np.var(images_plt, axis = 0)

im = ax[1, 3].contourf(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), cmap=cmap_well, extend='neither')
ax[1, 3].contour(grid_X, grid_Y, images_plt_average[0] * 50,\
            levels = np.arange(vmin, vmax, (vmax - vmin) / levels), colors='black', linewidths=0.5)
ax[1, 3].scatter(grid_X_flat, grid_Y_flat, s=7, marker='s', c='', linewidths=1, edgecolors='black')


rmse = np.sqrt((((images_plt_average[0] - realrefers[refer_index,0,:,:]) * 50) ** 2).mean())
ax[1, 3].annotate('RMSE:' + np.round(rmse, 1).astype(str), xy=(33, 56), fontsize=ftsize,\
            bbox={'facecolor':'white', 'edgecolor':'white'})

ax[1, 3].set_title('('+ abc_list[7] + ')' + '4500 (' + '$10^3$' + ') images')



# colorbar设置
cax = add_right_cax(ax[0, 3], pad=0.02, width=0.02)
cbar = fig.colorbar(im, cax=cax, ticks=locator, format=formatter)
cbar.ax.tick_params(labelsize=ftsize)


cax = add_right_cax(ax[1, 3], pad=0.02, width=0.02)
cbar = fig.colorbar(im, cax=cax, ticks=locator, format=formatter)
cbar.ax.tick_params(labelsize=ftsize)

plt.savefig(network_dir + "intermediate_img.png", dpi=600) 