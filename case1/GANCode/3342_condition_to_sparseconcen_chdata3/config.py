#----------------------------------------------------------------------------
# Convenience class that behaves exactly like dict(), but allows accessing
# the keys and values using the attribute syntax, i.e., "mydict.key = value".

class EasyDict(dict):
    def __init__(self, *args, **kwargs): super().__init__(*args, **kwargs)
    def __getattr__(self, name): return self[name]
    def __setattr__(self, name, value): self[name] = value
    def __delattr__(self, name): del self[name]

#----------------------------------------------------------------------------
# Paths.

data_dir = '/public/home/yhn/back3_test7/tfdataset'  # Training data path
# "data_dir" refers to the path of grandparent directory of training dataset like *.tfrecord files. "dataset" in line 46 refers to parent folder name of training dataset.
# e.g., folder "AA/BB/CC" includes all the *.tfrecord files training dataset, then data_dir = 'AA/BB/', and in line 46, tfrecord_dir=  'CC'. 

result_dir = '/public/home/yhn/back3_test7/GANresults/TrainingResults_3342_chdata3_lesswell'  # result data path

#----------------------------------------------------------------------------
# TensorFlow options.

tf_config = EasyDict()  # TensorFlow session config, set by tfutil.init_tf().
env = EasyDict()        # Environment variables, set by the main program in train.py.
tf_config['graph_options.place_pruned_graph']   = True      # False (default) = Check that all ops are available on the designated device. True = Skip the check for ops that are not used.
tf_config['gpu_options.allow_growth']          = True     # False (default) = Allocate all GPU memory at the beginning. True = Allocate only as much GPU memory as needed.
env.CUDA_VISIBLE_DEVICES                       = '0,1'       # Unspecified (default) = Use all available GPUs. List of ints = CUDA device numbers to use.
env.TF_CPP_MIN_LOG_LEVEL                        = '0'       # 0 (default) = Print all available debug info from TensorFlow. 1 = Print warnings and errors, but disable debug info.

#----------------------------------------------------------------------------
# To run, comment/uncomment the lines as appropriate and launch train.py.

desc        = 'pgan'                                        # Description string included in result subdir name.
random_seed = 1000                                          # Global random seed.
dataset     = EasyDict()                                    # Options for dataset.load_dataset().
train       = EasyDict(func='train.train_progressive_gan')  # Options for main training func.
G           = EasyDict(func='networks.G_paper')             # Options for generator network.
D           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
D2           = EasyDict(func='networks.D_paper')             # Options for discriminator network.
G_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for generator optimizer.
D_opt       = EasyDict(beta1=0.0, beta2=0.99, epsilon=1e-8) # Options for discriminator optimizer.
G_loss      = EasyDict(func='loss.G_wgan_acgan')            # Options for generator loss.
G_PDE_loss  = EasyDict(func='loss.G_PDE_wgan_acgan')        # Options for generator PDE loss.
D_loss      = EasyDict(func='loss.D_wgangp_acgan')          # Options for discriminator loss.
D2_loss      = EasyDict(func='loss.D2_wgangp_acgan')          # Options for discriminator loss.
sched       = EasyDict()                                    # Options for train.TrainingSchedule.
grid        = EasyDict(size='6by8')                         # Options for train.setup_snapshot_image_grid().

dataset = EasyDict(tfrecord_dir= 'TrainingData_3342_chdata3_lesswell')   #Replace 'TrainingData(MultiChannels_Version4)' with parent folder name of *.tfrecords training dataset.  


desc += '-2gpu'; num_gpus = 2; sched.minibatch_base = 32; 
sched.minibatch_dict = {4: 256, 8: 256, 16: 128, 32: 128, 64: 128}; 
sched.G_lrate_dict = {4: 0.005, 8: 0.01, 16: 0.01, 32: 0.007, 64: 0.005}; 
sched.D_lrate_dict = EasyDict(sched.G_lrate_dict); 
train.total_kimg = 4500
sched.max_minibatch_per_gpu = {32: 128, 64: 128}


sched.G_pdelossweight_dict = {4: 0, 8: 0, 16: 0, 32: 0, 64: 4}; 




sched.lod_training_kimg_dict  = {4: 160, 8:320, 16:320, 32:320, 64:320}

# ** Uncomment following one line of code if using conventional GAN training process. **
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; train.total_kimg = 10000

# Disable individual features.
#desc += '-nopixelnorm'; G.use_pixelnorm = False
#desc += '-nowscale'; G.use_wscale = False; D.use_wscale = False
#desc += '-noleakyrelu'; G.use_leakyrelu = False
#desc += '-nosmoothing'; train.G_smoothing = 0.0
#desc += '-norepeat'; train.minibatch_repeats = 1
#desc += '-noreset'; train.reset_opt_for_new_lod = False



#----------------------------------------------
# Settings for condition to well facies data
desc += '-CondWell';          
# dataset.well_enlarge = True; desc += '-Enlarg';  # uncomment this line to let the dataset output enlarged well facies data; comment to make it unenlarged.
dataset.max_label_size  = 'full'
#----------------------------------------------
# Setting if loss normalization (into standard Gaussian) is used 
# G_loss.lossnorm = True
#----------------------------------------------
# Set if no growing, i.e., the conventional training method. Can be used if only global features are conditioned.
#desc += '-nogrowing'; sched.lod_initial_resolution = 64; sched.lod_training_kimg = 0; sched.lod_transition_kimg = 0; train.total_kimg = 10000


sched.G_welllossweight_dict = {4: 1, 8: 1, 16: 1, 32: 1, 64: 1}; 

G_loss.Wellfaciesloss_weight = 1000
G_loss.label_weight = 10
G_loss.realrefersloss_weight=1
G_loss.orig_weight = 1

D_loss.label_weight = 1
D2_loss.label_weight = 100
# desc += '-labelpenalty';   # constrain of time label
# desc += '-realreferpenalty';  # constrain of t-1 concentration image

# desc += '-Time_weight10';


# desc += '-PDE_loss';
# G_loss.pde_weight=5
# G_loss.bound_weight=5
# G_loss.ini_weight=5

# # desc += '-vconsist';
# G_loss.vconsist_weight=0

# G_PDE_loss.pde_weight=5
# G_PDE_loss.bound_weight=5
# G_PDE_loss.ini_weight=5
# G_PDE_loss.vconsist_weight=5