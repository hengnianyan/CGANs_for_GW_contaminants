
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import numpy as np

import tensorflow as tf


class SobelFilter_tf(object):
    
    def __init__(self,imsize,delta_x,correct=True):
        # conv2d is cross-correlation, need to transpose the kernel here ???
        
        kernel_3x3_arr = np.reshape(np.array([[-1, -2, -1],
                                               [0, 0, 0],
                                                [1, 2, 1]])/(8.0*delta_x),[3,3,1,1])
        
        self.HSOBEL_WEIGHTS_3x3 = tf.constant(kernel_3x3_arr,dtype=tf.float32)
        self.VSOBEL_WEIGHTS_3x3 = tf.transpose(self.HSOBEL_WEIGHTS_3x3,perm=[1,0,2,3])
        
        kernel_5x5_arr = np.reshape(np.array([[-5, -4, 0, 4, 5],
                                            [-8, -10, 0, 10, 8],
                                            [-10, -20, 0, 20, 10],
                                            [-8, -10, 0, 10, 8],
                                            [-5, -4, 0, 4, 5]])/240.,[5,5,1,1])
        self.HSOBEL_WEIGHTS_5x5 = tf.constant(kernel_5x5_arr,dtype=tf.float32)
        self.VSOBEL_WEIGHTS_5x5 = tf.transpose(self.HSOBEL_WEIGHTS_5x5,perm=[1,0,2,3])
        
        modifier = np.eye(imsize)
        modifier[0:2,0] = np.array([4, -1])
        modifier[-2:,-1] = np.array([-1, 4])
        self.modifier = tf.constant(modifier,dtype = tf.float32)
        self.correct = correct
        self.imsize = imsize
    
    def pad_image(self,one_image): # x是[64,64,1]
        one_image = tf.squeeze(one_image) # 保证用于pad的是2维数据
        padded_image = tf.pad(one_image,[[1,1],[1,1]],'SYMMETRIC')
        return padded_image
    
    def grad_h(self, image, filter_size=3):
        """Get image gradient along horizontal direction, or x axis.
        Option to do replicate padding for image before convolution. This is mainly
        for estimate the du/dy, enforcing Neumann boundary condition.

        Args:
            images (Tensor): (N, H, W, 1)  需要改写成适用于(N, H, W, 1)
            replicate_pad (None, int, 4-tuple): if 4-tuple, (padLeft, padRight, padTop, 
                padBottom)
        
        Return:
            grad (N, H, W) 
        """
        # image_width = tf.cast(image.shape[-2],dtype=tf.float32)  ## tensorflow中的image tensor一般是 NHWC的数据格式
        # image_width = 64
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.VSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.VSOBEL_WEIGHTS_5x5
        
        padded_image = tf.expand_dims(tf.map_fn(self.pad_image,image),axis=-1)
        # grad = tf.nn.conv2d(padded_image,kernel,strides=[1,1,1,1],padding='VALID') * image_width
        grad = tf.nn.conv2d(padded_image,kernel,strides=[1,1,1,1],padding='VALID')
        
        if self.correct:
            result = tf.map_fn(lambda x: tf.matmul(tf.squeeze(x),self.modifier),grad)
            return tf.expand_dims(result,axis=-1)
        else:
            return grad
        
    
    def grad_v(self,image,filter_size=3):
        # image_height = tf.cast(image.shape[-3],dtype=tf.float32)
        # image_height = 64
        if filter_size == 3:
            replicate_pad = 1
            kernel = self.HSOBEL_WEIGHTS_3x3
        elif filter_size == 5:
            replicate_pad = 2
            kernel = self.HSOBEL_WEIGHTS_5x5
            
        padded_image = tf.expand_dims(tf.map_fn(self.pad_image,image),axis=-1)
        # grad = tf.nn.conv2d(padded_image,kernel,strides=[1,1,1,1],padding='VALID') * image_height
        grad = tf.nn.conv2d(padded_image,kernel,strides=[1,1,1,1],padding='VALID')


        if self.correct:
            result = tf.map_fn(lambda x: tf.matmul(tf.squeeze(x),self.modifier),grad)
            return tf.expand_dims(result,axis=-1)
        else:
            return grad