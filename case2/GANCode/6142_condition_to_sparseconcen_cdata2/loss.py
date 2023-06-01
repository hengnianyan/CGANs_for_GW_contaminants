
# import numpy as np
import tensorflow as tf
import tfutil
from SobelFilter import SobelFilter_tf
import math
# from multiprocessing import Pool

#----------------------------------------------------------------------------
# Convenience func that casts all of its arguments to tf.float32.

def fp32(*values):
    if len(values) == 1 and isinstance(values[0], tuple):
        values = values[0]
    values = tuple(tf.cast(v, tf.float32) for v in values)
    return values if len(values) >= 2 else values[0]




def G_wgan_acgan(G,D,  lod, labels, well_facies,  minibatch_size, well_facies3,
                    Wellfaciesloss_weight = 1, G_welllossweight=1,Wellsmoothloss_weight = 10,label_weight = 10, realrefersloss_weight=1, orig_weight = 1, labeltypes = None, 
                    pde_weight=1, bound_weight=5, ini_weight=5,G_pdelossweight=1,vconsist_weight=5):

    Wellfaciesloss_weight = Wellfaciesloss_weight * G_welllossweight

    labels_in = labels # labels_in在G中不会起作用
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    well_facies = tf.cast(well_facies, tf.float32)

    fake_out = G.get_output_for(latents, labels_in, well_facies, is_training=True) 

    fake_images_out = fake_out


    fake_scores_out, _ = fp32(D.get_output_for(fake_out, is_training=True))

    loss = -fake_scores_out
    loss = tfutil.autosummary('Loss_G/GANloss', loss)
    loss = loss * orig_weight     

    #### all observation
    fake_images_out3 = G.get_output_for(latents, labels_in, well_facies3, is_training=True)


    def Wellpoints_L2loss(well_facies, fake_images):
        loss = 2 * tf.nn.l2_loss(well_facies[:,0:1]* (well_facies[:,1:2] - (fake_images+1)/2))
        loss = loss / tf.reduce_sum(well_facies[:, 0:1])
        return loss

    def Wellpoints_liklihood_loss(well_facies, fake_images):
        # 对数似然函数

        err = tf.exp(well_facies[:,1:2] * 6) - tf.exp(((fake_images + 1) / 2) * 6)
        err = well_facies[:,0:1] * err

        N_grid = 64*64
        err = tf.transpose(tf.reshape(err,[-1,N_grid]))

        obs_var = (0.1)**2

        Cd_inv = 1/obs_var*tf.eye(N_grid) # 对角矩阵的逆
        loss_temp = tf.matmul(tf.matmul(tf.transpose(err),Cd_inv) , err)
        loss_temp = -1/2*tf.diag(loss_temp) # 只保留对角线


        Nd = tf.reduce_sum(tf.reshape(well_facies[:, 0:1],[-1,N_grid]),axis=1)
        coeff = tf.log(1.0) - (Nd/2)*tf.log((2*math.pi)) - 1/2*Nd*tf.log(obs_var)


        loss = tf.reduce_mean(-(coeff+loss_temp))

        return loss

    def addwellfaciespenalty(well_facies, fake_images_out, loss, Wellfaciesloss_weight):
        with tf.name_scope('WellfaciesPenalty'):
            WellfaciesPenalty =  Wellpoints_L2loss(well_facies, fake_images_out)   
            WellfaciesPenalty = tfutil.autosummary('Loss_G/WellfaciesPenalty', WellfaciesPenalty)
            loss += WellfaciesPenalty * Wellfaciesloss_weight   
        return loss   
    loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addwellfaciespenalty(well_facies, fake_images_out, loss, 100 * 0.5), lambda: loss)
    loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addwellfaciespenalty(well_facies3, fake_images_out3, loss, Wellfaciesloss_weight * 0.5), lambda: loss)

    # local variance
    def Wellpoints_smooth_loss(well_facies, fake_images):
        images_smooth = tf.nn.avg_pool(fake_images,ksize=[1,1,3,3], strides=[1,1,1,1], padding='SAME', data_format='NCHW')
        loss = tf.nn.l2_loss(well_facies[:,0:1]* (fake_images - images_smooth))
        loss = loss / tf.reduce_sum(well_facies[:, 0:1])
        return loss

    def addwellsmoothpenalty(well_facies, fake_images_out, loss, Wellsmoothloss_weight):
        with tf.name_scope('WellsmoothPenalty'):        
            WellsmoothPenalty =  Wellpoints_smooth_loss(well_facies, fake_images_out)
            WellsmoothPenalty = tfutil.autosummary('Loss_G/WellsmoothPenalty', WellsmoothPenalty)
            loss += WellsmoothPenalty * Wellsmoothloss_weight   
        return loss   
    # loss = tf.cond(tf.math.less_equal(lod, tf.fill([], 4.)), lambda: addwellsmoothpenalty(well_facies, fake_images_out, loss, Wellsmoothloss_weight), lambda: loss)


    fake_images_out = (fake_images_out+1)/2


    
    # resolution = 2 ** (resolution_log2 - int(np.floor(lod))) # resolution_log2代表原始图像大小. 64*64, 2^6, 6
    
    return loss



def G_PDE_wgan_acgan(G,  lod, labels, well_facies, velocity, minibatch_size,
                    pde_weight=1, bound_weight=5, ini_weight=5,vconsist_weight=5):
    
    labels_in = labels
    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    well_facies = tf.cast(well_facies, tf.float32)
    # realrefers = tf.cast(realrefers, tf.float32)


    fake_out = G.get_output_for(latents, labels_in, well_facies, is_training=True)

    velocity_fake = tf.exp(((fake_out[:,1:]+1)/2)*1.5+1)-5
    fake_images_out = fake_out[:,0:1]
    fake_images_out = (fake_images_out+1)/2
    
    # resolution = 2 ** (resolution_log2 - int(np.floor(lod))) # resolution_log2代表原始图像大小. 64*64, 2^6, 6

    # Dispersion coefficient D_h D_v
    disp = 5
    diffu = 0.01
    v_norm = tf.sqrt(tf.square(velocity_fake[:,0:1])+tf.square(velocity_fake[:,1:2]))  # magnitude of the velocity vector
    D_h = disp*tf.square(velocity_fake[:,0:1])/v_norm + \
        0.1*disp*tf.square(velocity_fake[:,1:2])/v_norm + \
        diffu

    D_v = disp*tf.square(velocity_fake[:,1:2])/v_norm + \
        0.1*disp*tf.square(velocity_fake[:,0:1])/v_norm + \
        diffu

    # 去除边界上的nan值，v_norm很小时出现
    D_h = tf.where(tf.is_nan(D_h), diffu*tf.ones_like(D_h), D_h)
    D_v = tf.where(tf.is_nan(D_v), diffu*tf.ones_like(D_v), D_v)
    D_hv = (disp-0.1*disp)*((velocity_fake[:,0:1]*velocity_fake[:,1:2])/v_norm)
    D_hv = tf.where(tf.is_nan(D_hv), tf.zeros_like(D_hv), D_hv)

    # 本案例计算梯度前不需要去归一化
    

    # 差分法
    delta_t = 2.5
    delta_x = 1.5625
    labels_in2 = labels_in - delta_t/500
    
    part1 = well_facies[:,0:2,:,:]
    part1 = tf.zeros_like(part1) #去掉所有condition点
    part2 = well_facies[:,2:,:,:]
    well_facies = tf.concat([part1,part2],1)


    fake_images_out2 = G.get_output_for(latents, labels_in2, well_facies, is_training=True)
    # velocity_fake2 = tf.exp(((fake_images_out2[:,1:]+1)/2)*1.5+3)-35.5
    velocity_fake2 = tf.exp(((fake_images_out2[:,1:]+1)/2)*1.5+1)-5
    fake_images_out2 = fake_images_out2[:,0:1]
    fake_images_out2 = (fake_images_out2+1)/2

    Ct = (fake_images_out2 - fake_images_out)/delta_t

    # space derivative
    resolution = fake_images_out.shape[-1]
    sobel_filter = SobelFilter_tf(resolution,delta_x)

    #临时测试
    # fake_images_out = well_facies[:,2:3]

    # tf.transpose(sobel_filter.grad_h(tf.transpose(,[0,2,3,1])) , [0,3,1,2])
    grad_h = tf.transpose(sobel_filter.grad_h(tf.transpose(fake_images_out,[0,2,3,1])) , [0,3,1,2])  # grad_h操作的数据是(N, H, W, 1) , x direction
    grad_v = tf.transpose(sobel_filter.grad_v(tf.transpose(fake_images_out,[0,2,3,1])) , [0,3,1,2])
    # grad_v = sobel_filter.grad_v(fake_images_out)

    grad_hh = tf.transpose(sobel_filter.grad_h(tf.transpose(D_h*grad_h,[0,2,3,1])) , [0,3,1,2]) # 弥散项
    grad_hv = tf.transpose(sobel_filter.grad_h(tf.transpose(D_hv*grad_v,[0,2,3,1])) , [0,3,1,2])
    grad_vv = tf.transpose(sobel_filter.grad_v(tf.transpose(D_v*grad_v,[0,2,3,1])) , [0,3,1,2])
    grad_vh = tf.transpose(sobel_filter.grad_v(tf.transpose(D_hv*grad_h,[0,2,3,1])) , [0,3,1,2])
    
    grad_h2 = tf.transpose(sobel_filter.grad_h(tf.transpose(velocity_fake[:,0:1,:,:]*fake_images_out,[0,2,3,1])) , [0,3,1,2]) # 对流项
    grad_v2 = tf.transpose(sobel_filter.grad_v(tf.transpose(velocity_fake[:,1:2,:,:]*fake_images_out,[0,2,3,1])) , [0,3,1,2])

    pde_loss = tf.reduce_mean(tf.square(Ct - grad_hh -grad_hv - grad_vv - grad_vh + grad_h2 +grad_v2))

    # boundary constrain
    # right_bound = fake_images_out[:,:,:,-1]
    # bound_loss = tf.reduce_mean(tf.square(tf.zeros_like(right_bound)-right_bound))
    # bound_loss = tf.nn.l2_loss(tf.zeros_like(right_bound)-right_bound)

    # initial constrain
    labels_in_ini = tf.zeros_like(labels_in)
    fake_images_out = G.get_output_for(latents, labels_in_ini, well_facies, is_training=True)
    # velocity_fake_ini = tf.exp(((fake_images_out[:,1:]+1)/2)*1.5+3)-35.5
    velocity_fake_ini = tf.exp(((fake_images_out[:,1:]+1)/2)*1.5+1)-5
    fake_images_out = fake_images_out[:,0:1]
    ini_loss = tf.reduce_mean(tf.square(fake_images_out - well_facies[:,2:3]))
    # ini_loss = tf.nn.l2_loss(fake_images_out - well_facies[:,2:3])
    
    # velocity consistent constrain
    vconsist_loss = tf.reduce_mean((tf.square(velocity_fake2-velocity_fake) + tf.square(velocity_fake_ini-velocity_fake) + tf.square(velocity_fake2-velocity_fake_ini))/3)


    # loss =  pde_weight*pde_loss + bound_weight*bound_loss + ini_weight*ini_loss + vconsist_weight*vconsist_loss
    loss =  pde_weight*pde_loss + ini_weight*ini_loss + vconsist_weight*vconsist_loss
    # loss =  bound_weight*bound_loss + ini_weight*ini_loss
    loss = tfutil.autosummary('Loss_G/PDE_loss', loss)

    loss_v = tf.reduce_mean(tf.square(velocity_fake-velocity))

    return loss, loss_v, vconsist_loss


#----------------------------------------------------------------------------
# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D_wgangp_acgan(G, D,  opt,  minibatch_size, reals, labels, well_facies, 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.    
    label_weight     = 1):      

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])

    fake_images_out = G.get_output_for(latents, labels, well_facies, is_training=True)
   
    reals = reals[:,1:2]

    real_scores_out, _ = fp32(D.get_output_for(reals, is_training=True))

    fake_scores_out, _ = fp32(D.get_output_for(fake_images_out, is_training=True))


    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss_D/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss_D/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon



    loss = tfutil.autosummary('Loss_D/Total_loss', loss)

    return loss


def D3_wgangp_acgan(G, D,  opt,  minibatch_size, reals, labels, well_facies, 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.    
    label_weight     = 1):      

    latents = tf.random_normal([minibatch_size] + G.input_shapes[0][1:])
    # fake_images_out = G.get_output_for(latents, labels, is_training=True)
    fake_images_out = G.get_output_for(latents, labels, well_facies, is_training=True)


    fake_images_out = fake_images_out[:,1:2]
    reals = reals[:,1:2]

    # real_scores_out, realrefers_labels_out = fp32(D.get_output_for(reals, is_training=True))
    real_scores_out, _ = fp32(D.get_output_for(reals, is_training=True))

    fake_scores_out, _ = fp32(D.get_output_for(fake_images_out, is_training=True))


    real_scores_out = tfutil.autosummary('Loss_D/real_scores', real_scores_out)
    fake_scores_out = tfutil.autosummary('Loss_D/fake_scores', fake_scores_out)
    loss = fake_scores_out - real_scores_out

    with tf.name_scope('GradientPenalty'):
        mixing_factors = tf.random_uniform([minibatch_size, 1, 1, 1], 0.0, 1.0, dtype=fake_images_out.dtype)
        mixed_images_out = tfutil.lerp(tf.cast(reals, fake_images_out.dtype), fake_images_out, mixing_factors)
        mixed_scores_out, mixed_labels_out = fp32(D.get_output_for(mixed_images_out, is_training=True))
        mixed_scores_out = tfutil.autosummary('Loss_D/mixed_scores', mixed_scores_out)
        mixed_loss = opt.apply_loss_scaling(tf.reduce_sum(mixed_scores_out))
        mixed_grads = opt.undo_loss_scaling(fp32(tf.gradients(mixed_loss, [mixed_images_out])[0]))
        mixed_norms = tf.sqrt(tf.reduce_sum(tf.square(mixed_grads), axis=[1,2,3]))
        mixed_norms = tfutil.autosummary('Loss_D/mixed_norms', mixed_norms)
        gradient_penalty = tf.square(mixed_norms - wgan_target)
    loss += gradient_penalty * (wgan_lambda / (wgan_target**2))
    loss = tfutil.autosummary('Loss_D/WGAN_GP_loss', loss)

    with tf.name_scope('EpsilonPenalty'):
        epsilon_penalty = tfutil.autosummary('Loss_D/epsilon_penalty', tf.square(real_scores_out))
    loss += epsilon_penalty * wgan_epsilon



    loss = tfutil.autosummary('Loss_D/Total_loss', loss)

    return loss

# Discriminator loss function used in the paper (WGAN-GP + AC-GAN).

def D2_wgangp_acgan(G, D2, minibatch_size, labels, well_facies, reals, 
    wgan_lambda     = 10.0,     # Weight for the gradient penalty term.
    wgan_epsilon    = 0.001,    # Weight for the epsilon term, \epsilon_{drift}.
    wgan_target     = 1.0,      # Target value for gradient magnitudes.    
    label_weight     = 1):      


    reals = reals[:, 1:2]
    _,realrefers_labels_out = fp32(D2.get_output_for(reals, is_training=True))
    # fake_labels_out,_ = fp32(D2.get_output_for(fake_images_out, is_training=True))

    with tf.name_scope('LabelPenalty'):
        label_penalty_reals = tf.reduce_mean(tf.square(labels - realrefers_labels_out) )                    

        label_penalty_reals = tfutil.autosummary('Loss_D/label_penalty_reals', label_penalty_reals)

    loss = (label_penalty_reals ) * label_weight



    loss = tfutil.autosummary('Loss_D/Total_loss', loss)

    return loss