"""
Tutorial Machine Learning in Solid Mechanics (WiSe 22/23)
Task 1: Feed-Forward Neural Networks

==================

Authors: Dominik K. Klein
         
08/2022
"""

# %%   
"""
Import modules

"""

import tensorflow as tf
import numpy as np


# %%   
"""
Generate data for a bathtub function

"""

def bathtub() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:

    xs = np.linspace(1,10,450)
    ys = np.concatenate([np.square(xs[0:150]-4)+1, \
                         1+0.1*np.sin(np.linspace(0,3.14,90)), np.ones(60), \
                         np.square(xs[300:450]-7)+1])
    
        
    xs = xs / 10.0
    ys = ys / 10.0

    xs_c = np.concatenate([xs[0:240], xs[330:420]])
    ys_c = np.concatenate([ys[0:240], ys[330:420]])

    xs = tf.expand_dims(xs, axis = 1)
    ys = tf.expand_dims(ys, axis = 1)

    xs_c = tf.expand_dims(xs_c, axis = 1)
    ys_c = tf.expand_dims(ys_c, axis = 1)
    
    return xs, ys, xs_c, ys_c



# %%   
"""
Generate data for the 2D functions

"""

def get_f1() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
    x, y = x.ravel(), y.ravel() 
    f1 = x**2 - y**2
    
    x = tf.expand_dims(x, axis = 1)
    y = tf.expand_dims(y, axis = 1)
    f1 = tf.expand_dims(f1, axis = 1)
    
    return x, y, f1


def get_f2__delf2_delx() -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
    x, y = np.meshgrid(np.linspace(-4, 4, 20), np.linspace(-4, 4, 20))
    x, y = x.ravel(), y.ravel()
    f2 = x**2 + 0.5*y**2
    jac_f2 = np.array([2*x, y])
    
    x = tf.expand_dims(x, axis = 1)
    y = tf.expand_dims(y, axis = 1)
    f2 = tf.expand_dims(f2, axis = 1)
    jac_f2 = tf.transpose(jac_f2)
    
    return x, y, f2, jac_f2