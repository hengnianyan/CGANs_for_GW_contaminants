
import os
import time
import numpy as np
import tensorflow as tf

import config
import tfutil
import dataset
import misc

#----------------------------------------------------------------------------
# Choose the size and contents of the image snapshot grids that are exported
# periodically during training.

def setup_snapshot_image_grid(G, training_set, size = '6by8'):       # '6by8'=6row and 8 column.

    gw = 8
    gh = 6

    # Fill in reals and labels.
    reals = np.zeros([gw * gh] + training_set.shape, dtype=training_set.dtype)
    labels = np.zeros([gw * gh, training_set.label_size], dtype=training_set.label_dtype)
    wellfacies = np.zeros([gw * gh] + [2,64,64], dtype=np.float32)

    realrefers = np.zeros([gw * gh] + [2,64,64], dtype=np.float32)

    for idx in range(gw * gh):
        x = idx % gw; y = idx // gw
        while True:
            real, label = training_set.get_minibatch_np(1)
            wellface, refer, _ = training_set.get_minibatch_well_np(1)

            reals[idx] = real[0]
            labels[idx] = label[0]
            wellfacies[idx] = wellface[0]

            realrefers[idx] = refer[0]


            break

    # Generate latents.
    latents = misc.random_latents(gw * gh, G)
    return (gw, gh), reals, labels, wellfacies, realrefers, latents


#----------------------------------------------------------------------------
# Just-in-time processing of training images before feeding them to the networks.

def process_reals(x, lod, mirror_augment, drange_data, drange_net):
    with tf.name_scope('ProcessReals'):

        with tf.name_scope('DynamicRange'):
            x = tf.cast(x, tf.float32)
            x = misc.adjust_dynamic_range(x, drange_data, drange_net)
            

        if mirror_augment:
            with tf.name_scope('MirrorAugment'):
                s = tf.shape(x)
                mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
                mask = tf.tile(mask, [1, s[1], s[2], s[3]])
                x = tf.where(mask < 0.5, x, tf.reverse(x, axis=[3]))
        with tf.name_scope('FadeLOD'): # Smooth crossfade between consecutive levels-of-detail.
            s = tf.shape(x)
            y = tf.reshape(x, [-1, s[1], s[2]//2, 2, s[3]//2, 2])
            y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
            y = tf.tile(y, [1, 1, 1, 2, 1, 2])
            y = tf.reshape(y, [-1, s[1], s[2], s[3]])
            x = tfutil.lerp(x, y, lod - tf.floor(lod))
        with tf.name_scope('UpscaleLOD'): # Upscale to match the expected input/output size of the networks.
            s = tf.shape(x)
            factor = tf.cast(2 ** tf.floor(lod), tf.int32)
            x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
            x = tf.tile(x, [1, 1, 1, factor, 1, factor])
            x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
        return x

#----------------------------------------------------------------------------
# Class for evaluating and storing the values of time-varying training parameters.

class TrainingSchedule:
    def __init__(
        self,
        cur_nimg,
        training_set,
        lod_initial_resolution  = 4,        # Image resolution used at the beginning.
        lod_training_kimg_dict  = {4: 160, 8:160, 16:160, 32:160, 64:160},# Thousands of real images to show before doubling the resolution.
        lod_transition_kimg_dict= {4: 160, 8:160, 16:160, 32:160, 64:160},      # Thousands of real images to show when fading in new layers.
        minibatch_base          = 16,       # Maximum minibatch size, divided evenly among GPUs.
        minibatch_dict          = {},       # Resolution-specific overrides.
        max_minibatch_per_gpu   = {},       # Resolution-specific maximum minibatch size per GPU.
        G_lrate_base            = 0.001,    # Learning rate for the generator.
        G_lrate_dict            = {},       # Resolution-specific overrides.
        D_lrate_base            = 0.001,    # Learning rate for the discriminator.
        D_lrate_dict            = {},       # Resolution-specific overrides.
        G_pdelossweight_dict    = {},
        G_pdelossweight_base    = 1,
        G_welllossweight_dict    = {},
        G_welllossweight_base    = 1,
        tick_kimg_base          = 80,      # Default interval of progress snapshots.
        tick_kimg_dict          = {4: 160, 8:160, 16:160, 32:160, 64:160}): # Resolution-specific overrides.

        # Training phase.
        self.kimg = cur_nimg / 1000.0
        
        train_kimg_sum = 0
        trans_kimg_sum = 0 
        for i in range(5):
            train_list = list(lod_training_kimg_dict.values())
            trans_list = list(lod_training_kimg_dict.values())
            train_kimg_sum += train_list[i]
            trans_kimg_sum += trans_list[i]

            if train_kimg_sum + trans_kimg_sum > self.kimg: 
                phase_idx = i
                lod_training_kimg = train_list[i]
                lod_transition_kimg = trans_list[i]
                break
            phase_idx = i
            lod_training_kimg = train_list[i]
            lod_transition_kimg = trans_list[i]
        phase_kimg = self.kimg - ((train_kimg_sum - train_list[phase_idx]) + (trans_kimg_sum - trans_list[i]))

        # Level-of-detail and resolution.
        self.lod = training_set.resolution_log2
        self.lod -= np.floor(np.log2(lod_initial_resolution))
        self.lod -= phase_idx
        if lod_transition_kimg > 0:
            self.lod -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        self.lod = max(self.lod, 0.0)
        self.resolution = 2 ** (training_set.resolution_log2 - int(np.floor(self.lod)))

        # Minibatch size.
        self.minibatch = minibatch_dict.get(self.resolution, minibatch_base)
        self.minibatch -= self.minibatch % config.num_gpus
        if self.resolution in max_minibatch_per_gpu:
            self.minibatch = min(self.minibatch, max_minibatch_per_gpu[self.resolution] * config.num_gpus)

        # Other parameters.
        self.G_lrate = G_lrate_dict.get(self.resolution, G_lrate_base)
        self.D_lrate = D_lrate_dict.get(self.resolution, D_lrate_base)
        self.tick_kimg = tick_kimg_dict.get(self.resolution, tick_kimg_base)
        self.G_pdelossweight = G_pdelossweight_dict.get(self.resolution, G_pdelossweight_base)
        self.G_welllossweight = G_welllossweight_dict.get(self.resolution, G_welllossweight_base)

#----------------------------------------------------------------------------
# Main training script.
# To run, comment/uncomment appropriate lines in config.py and launch train.py.

def train_progressive_gan(
    G_smoothing             = 0.999,        # Exponential running average of generator weights.
    D_repeats               = 1,            # How many times the discriminator is trained per G iteration.
    minibatch_repeats       = 4,            # Number of minibatches to run before adjusting training parameters.
    reset_opt_for_new_lod   = True,         # Reset optimizer internal state (e.g. Adam moments) when new layers are introduced?
    total_kimg              = 15000,        # Total length of the training, measured in thousands of real images.
    mirror_augment          = False,        # Enable mirror augment?
    drange_net              = [-1,1],       # Dynamic range used when feeding image data to the networks.
    image_snapshot_ticks    = 1,            # How often to export image snapshots?
    network_snapshot_ticks  = 1,           # How often to export network snapshots?
    save_tf_graph           = False,        # Include full TensorFlow computation graph in the tfevents file?
    save_weight_histograms  = False,        # Include weight histograms in the tfevents file?
    resume_run_id           = None, #'/scratch/users/suihong/ProGAN_MultiChannel_Reusults_ConditionedtoMultiConditions_TF/086-pgan-unconditional-follow78-preset-v2-2gpu-fp32/network-snapshot-014880.pkl', #         # Run ID or network pkl to resume training from, None = start from scratch.
    resume_snapshot         = None,         # Snapshot index to resume training from, None = autodetect.
    resume_kimg             = 0., #4240.0,          # Assumed training progress at the beginning. Affects reporting and training schedule.
    resume_time             = 0.): # 2*3600+34*60+29        # Assumed wallclock time at the beginning. Affects reporting.

    maintenance_start_time = time.time()
    training_set = dataset.load_dataset(data_dir=config.data_dir, verbose=True, **config.dataset)
    print(training_set.label_size)

    # Construct networks.
    with tf.device('/gpu:0'):
        if resume_run_id is not None:
            network_pkl = misc.locate_network_pkl(resume_run_id, resume_snapshot)
            print('Loading networks from "%s"...' % network_pkl)
            G, D, Gs = misc.load_pkl(network_pkl)
        else:
            print('Constructing networks...')
            G = tfutil.Network('G', num_channels=1, resolution=training_set.shape[1], label_size=training_set.label_size, **config.G)
            D = tfutil.Network('D', num_channels=1, resolution=training_set.shape[1], label_size=0, **config.D) # C
            D2 = tfutil.Network('D2', num_channels=1, resolution=training_set.shape[1], label_size=training_set.label_size, **config.D2) # input two concentration image
            Gs = G.clone('Gs')
        Gs_update_op = Gs.setup_as_moving_average_of(G, beta=G_smoothing)
    G.print_layers(); D.print_layers()

    print('Building TensorFlow graph...')
    with tf.name_scope('Inputs'):
        lod_in          = tf.placeholder(tf.float32, name='lod_in', shape=[])
        lrate_in        = tf.placeholder(tf.float32, name='lrate_in', shape=[])
        minibatch_in    = tf.placeholder(tf.int32, name='minibatch_in', shape=[])
        G_pdelossweight_in = tf.placeholder(tf.float32, name='G_pdelossweight_in', shape=[])
        G_welllossweight_in = tf.placeholder(tf.float32, name='G_welllossweight_in', shape=[])
        minibatch_split = minibatch_in // config.num_gpus
        reals, labels  = training_set.get_minibatch_tf(minibatch_in)

        wellfacies, realrefers, wellfacies3 = training_set.get_minibatch_well_tf(minibatch_in)
        wellfacies = tf.cast(wellfacies, tf.float32)

        wellfacies3 = tf.cast(wellfacies3, tf.float32)

        labels = tf.cast(labels, tf.float32)

        welllocs2 = tf.cast((wellfacies[:, 1:2] > 0.1), tf.float32)  # C
        wellfacies_corrected2 = (wellfacies[:, 1:2] - 1) * welllocs2   
 
        wellfacies = tf.concat([welllocs2, wellfacies_corrected2], 1)
        reals_split     = tf.split(reals, config.num_gpus)
        labels_split    = tf.split(labels, config.num_gpus)
        wellfacies_split    = tf.split(wellfacies, config.num_gpus)

        wellfacies3_split = tf.split(wellfacies3, config.num_gpus)

    G_opt = tfutil.Optimizer(name='TrainG', learning_rate=lrate_in, **config.G_opt)

    D_opt = tfutil.Optimizer(name='TrainD', learning_rate=lrate_in, **config.D_opt)

    D2_opt = tfutil.Optimizer(name='TrainD2', learning_rate=lrate_in, **config.D_opt)
    for gpu in range(config.num_gpus):
        with tf.name_scope('GPU%d' % gpu), tf.device('/gpu:%d' % gpu):
            G_gpu = G if gpu == 0 else G.clone(G.name + '_shadow')
            D_gpu = D if gpu == 0 else D.clone(D.name + '_shadow')

            D2_gpu = D2 if gpu == 0 else D2.clone(D2.name + '_shadow')

            lod_assign_ops = [tf.assign(G_gpu.find_var('lod'), lod_in), tf.assign(D_gpu.find_var('lod'), lod_in), tf.assign(D2_gpu.find_var('lod'), lod_in)]
            reals_gpu = process_reals(reals_split[gpu], lod_in, mirror_augment, training_set.dynamic_range, drange_net)
            labels_gpu = labels_split[gpu]
            wellfacies_gpu = wellfacies_split[gpu]
            wellfacies3_gpu = wellfacies3_split[gpu]

            with tf.name_scope('G_loss'), tf.control_dependencies(lod_assign_ops):
                G_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu,  lod = lod_in, labels = labels_gpu, well_facies = wellfacies_gpu, minibatch_size=minibatch_split, well_facies3=wellfacies3_gpu, G_welllossweight=G_welllossweight_in, **config.G_loss)
            with tf.name_scope('D_loss'), tf.control_dependencies(lod_assign_ops):
                D_loss = tfutil.call_func_by_name(G=G_gpu, D=D_gpu, opt=D_opt, minibatch_size=minibatch_split, reals=reals_gpu, labels=labels_gpu, well_facies = wellfacies_gpu, **config.D_loss)
            with tf.name_scope('D2_loss'), tf.control_dependencies(lod_assign_ops):
                D2_loss = tfutil.call_func_by_name(G=G_gpu, D2=D2_gpu, minibatch_size=minibatch_split, labels=labels_gpu, well_facies = wellfacies_gpu, reals=reals_gpu, **config.D2_loss)

            
            G_opt.register_gradients(tf.reduce_mean(G_loss), G_gpu.trainables)
            D_opt.register_gradients(tf.reduce_mean(D_loss), D_gpu.trainables)
            D2_opt.register_gradients(tf.reduce_mean(D2_loss), D2_gpu.trainables)
    G_train_op = G_opt.apply_updates()
    D_train_op = D_opt.apply_updates()
    D2_train_op = D2_opt.apply_updates()

    print('Setting up snapshot image grid...')
    grid_size, grid_reals, grid_labels, grid_wellfacies,grid_realrefers, grid_latents = setup_snapshot_image_grid(G, training_set, **config.grid)    
    sched = TrainingSchedule(total_kimg * 1000, training_set, **config.sched)    
    grid_wellfacies_process = np.concatenate(((grid_wellfacies[:, 1:2] > 0),  (grid_wellfacies[:, 1:2] - 1) * (grid_wellfacies[:, 1:2] > 0)) , 1)
    grid_fakes = Gs.run(grid_latents, grid_labels, grid_wellfacies_process, minibatch_size=sched.minibatch//config.num_gpus)


    

    print('Setting up result dir...')
    result_subdir = misc.create_result_subdir(config.result_dir, config.desc)

    misc.save_image_grid(grid_wellfacies[:, 0:1]/2, os.path.join(result_subdir, 'wellfacies_c.png'), drange=[0, 1], grid_size=grid_size)
    misc.save_image_grid(grid_wellfacies[:, 1:2]/2, os.path.join(result_subdir, 'wellfacies_c_sorp.png'), drange=[0, 1], grid_size=grid_size)

    misc.save_image_grid(grid_realrefers[:, 0:1], os.path.join(result_subdir, 'realrefers_c.png'), drange=[0, 1], grid_size=grid_size)
    misc.save_image_grid(grid_realrefers[:, 1:2], os.path.join(result_subdir, 'realrefers_c_sorp.png'), drange=[0, 1], grid_size=grid_size)

    misc.save_image_grid(grid_fakes[:, 0:1], os.path.join(result_subdir, 'fakes_c_sorp%06d.png' % 0), drange=[-1, 1], grid_size=grid_size)

    summary_log = tf.summary.FileWriter(result_subdir)
    if save_tf_graph:
        summary_log.add_graph(tf.get_default_graph())
    if save_weight_histograms:
        G.setup_weight_histograms(); D.setup_weight_histograms()

    print('Training...')
    cur_nimg = int(resume_kimg * 1000)
    cur_tick = 0
    tick_start_nimg = cur_nimg
    tick_start_time = time.time()
    train_start_time = tick_start_time - resume_time
    prev_lod = -1.0


    while cur_nimg < total_kimg * 1000:

        # Choose training parameters and configure training ops.
        sched = TrainingSchedule(cur_nimg, training_set, **config.sched)
        training_set.configure(sched.minibatch, sched.lod)
        if reset_opt_for_new_lod:
            if np.floor(sched.lod) != np.floor(prev_lod) or np.ceil(sched.lod) != np.ceil(prev_lod):
                G_opt.reset_optimizer_state(); 

                D_opt.reset_optimizer_state()

                D2_opt.reset_optimizer_state()
        prev_lod = sched.lod

        # Loss report
        if cur_nimg == 0:
            [D_loss_out] = tfutil.run([tf.reduce_mean(D_loss)], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})

            [D2_loss_out] = tfutil.run([tf.reduce_mean(D2_loss)], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
            [G_loss_out] = tfutil.run([tf.reduce_mean(G_loss)], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch, \
                        G_pdelossweight_in: sched.G_pdelossweight,G_welllossweight_in: sched.G_welllossweight})

            print('D_loss %.4f D2_loss %.4f G_loss %.4f' % (D_loss_out, D2_loss_out, G_loss_out))

        # Run training ops.
        for repeat in range(minibatch_repeats):
            for _ in range(D_repeats):
                tfutil.run([D_train_op, Gs_update_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})

                tfutil.run([D2_train_op], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
                cur_nimg += sched.minibatch                
            tfutil.run([G_train_op], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch,\
                G_pdelossweight_in: sched.G_pdelossweight,G_welllossweight_in: sched.G_welllossweight})

            
        # Perform maintenance tasks once per tick.
        done = (cur_nimg >= total_kimg * 1000)
        if cur_nimg >= tick_start_nimg + sched.tick_kimg * 1000 or done:
            cur_tick += 1
            cur_time = time.time()
            tick_kimg = (cur_nimg - tick_start_nimg) / 1000.0
            tick_start_nimg = cur_nimg
            tick_time = cur_time - tick_start_time
            total_time = cur_time - train_start_time
            maintenance_time = tick_start_time - maintenance_start_time
            maintenance_start_time = cur_time
            
            # Loss report
            [D_loss_out] = tfutil.run([tf.reduce_mean(D_loss)], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
 
            [D2_loss_out] = tfutil.run([tf.reduce_mean(D2_loss)], {lod_in: sched.lod, lrate_in: sched.D_lrate, minibatch_in: sched.minibatch})
            [G_loss_out] = tfutil.run([tf.reduce_mean(G_loss)], {lod_in: sched.lod, lrate_in: sched.G_lrate, minibatch_in: sched.minibatch,\
                G_pdelossweight_in: sched.G_pdelossweight,G_welllossweight_in: sched.G_welllossweight})


            # Report progress.
            print('tick %-5d kimg %-8.1f lod %-5.2f minibatch %-4d time %-12s sec/tick %-7.1f sec/kimg %-7.2f maintenance %.1f' % (
                tfutil.autosummary('Progress/tick', cur_tick),
                tfutil.autosummary('Progress/kimg', cur_nimg / 1000.0),
                tfutil.autosummary('Progress/lod', sched.lod),
                tfutil.autosummary('Progress/minibatch', sched.minibatch),
                misc.format_time(tfutil.autosummary('Timing/total_sec', total_time)),
                tfutil.autosummary('Timing/sec_per_tick', tick_time),
                tfutil.autosummary('Timing/sec_per_kimg', tick_time / tick_kimg),
                tfutil.autosummary('Timing/maintenance_sec', maintenance_time)))

            # print('D_loss %.4f D2_loss %.4f G_loss %.4f G_PDE_loss %.4f V_loss %.4f  Vconsist %.4f' % (D_loss_out,D2_loss_out,G_loss_out,G_PDE_loss_out,V_loss_out,Vconsist_out))
            print('D_loss %.4f D2_loss %.4f G_loss %.4f' % (D_loss_out, D2_loss_out, G_loss_out))

            tfutil.autosummary('Timing/total_hours', total_time / (60.0 * 60.0))
            tfutil.autosummary('Timing/total_days', total_time / (24.0 * 60.0 * 60.0))
            tfutil.save_summaries(summary_log, cur_nimg)

            # Save snapshots.
            if cur_tick % image_snapshot_ticks == 0 or done:
                grid_fakes = G.run(grid_latents, grid_labels, grid_wellfacies_process, minibatch_size=sched.minibatch//config.num_gpus)

                misc.save_image_grid(grid_fakes[:, 0:1], os.path.join(result_subdir, 'fakes_c_sorp%06d.png' % (cur_nimg // 1000)), drange=drange_net, grid_size=grid_size)

 
            if cur_tick % network_snapshot_ticks == 0 or done:
                misc.save_pkl((G, D,D2, Gs), os.path.join(result_subdir, 'network-snapshot-%06d.pkl' % (cur_nimg // 1000)))

            # Record start time of the next tick.
            tick_start_time = time.time()

    # Write final results.
    misc.save_pkl((G, D,D2, Gs), os.path.join(result_subdir, 'network-final.pkl'))

    summary_log.close()
    open(os.path.join(result_subdir, '_training-done.txt'), 'wt').close()

#----------------------------------------------------------------------------
# Main entry point.
# Calls the function indicated in config.py.

if __name__ == "__main__":
    misc.init_output_logging()
    np.random.seed(config.random_seed)
    print('Initializing TensorFlow...')
    os.environ.update(config.env)
    tfutil.init_tf(config.tf_config)
    print('Running %s()...' % config.train['func'])
    tfutil.call_func_by_name(**config.train)
    print('Exiting...')