import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
import os
import time

sample_cnt = 100
save_counter = 1000
batch_size = 1000
save_dir = "experiment/batch_%s" % (batch_size)

def generator(input, is_train=True, is_reuse=False):
    n_units = 500
    with tf.variable_scope("generator", reuse=is_reuse):
        tl.layers.set_name_reuse(is_reuse)
        net_in = InputLayer(input, name='generator/in')
        net_l1 = DenseLayer(net_in, n_units=n_units, act=tf.nn.elu, name='generator/d1')
        net_l2 = DenseLayer(net_l1, n_units=n_units, act=tf.nn.sigmoid, name='generator/d2')
        net_l3 = DenseLayer(net_l2, n_units=sample_cnt, act=tl.activation.identity, name='generator/d3')
    return net_l3

def discriminator(input, is_train=True, is_reuse=False):
    n_units = 500
    with tf.variable_scope("discriminator", reuse=is_reuse):
        tl.layers.set_name_reuse(is_reuse)
        net_in = InputLayer(input, name='discriminator/in')
        net_l1 = DenseLayer(net_in, n_units=n_units, act=tf.nn.elu, name='discriminator/d1')
        net_l2 = DenseLayer(net_l1, n_units=n_units, act=tf.nn.elu, name='discriminator/d2')
        net_l3 = DenseLayer(net_l2, n_units=2, act=tl.activation.identity, name='discriminator/d3')
    return net_l3

def stats(d):
    return [np.mean(d), np.std(d)]

z = tf.placeholder(tf.float32, [batch_size, sample_cnt], name='noise')
true_data = tf.placeholder(tf.float32, [batch_size, sample_cnt], name='true_image')

net_g = generator(z)
net_d2 = discriminator(true_data)
net_d = discriminator(net_g.outputs, is_reuse=True)
# net_g2 = generator(z, is_train=False, is_reuse=True)

g_mean = tf.reduce_mean(net_g.outputs)
g_2 = tf.multiply(net_g.outputs, net_g.outputs)
g_2_mean = tf.reduce_mean(g_2)
g_var = g_2_mean-g_mean*g_mean

t_mean = tf.reduce_mean(true_data)
t_2 = tf.multiply(true_data, true_data)
t_2_mean = tf.reduce_mean(t_2)
t_var = t_2_mean-t_mean*t_mean

d_loss_real = tl.cost.cross_entropy(net_d2.outputs, tf.ones([batch_size], dtype=tf.int32), name='dreal')
d_loss_fake = tl.cost.cross_entropy(net_d.outputs, tf.zeros([batch_size], dtype=tf.int32), name='dfake')
d_loss = d_loss_fake + d_loss_real
g_loss = tl.cost.cross_entropy(net_d.outputs, tf.ones([batch_size], dtype=tf.int32), name='gfake')

g_vars = tl.layers.get_variables_with_name('generator', True, True)
d_vars = tl.layers.get_variables_with_name('discriminator', True, True)
net_g.print_params(False)
print("---------------")
net_d.print_params(False)

lr = tf.Variable(2e-4, dtype=tf.float32)
d_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(d_loss, var_list=d_vars)
g_optim = tf.train.AdamOptimizer(learning_rate=lr).minimize(g_loss, var_list=g_vars)

sess = tf.InteractiveSession()
tl.layers.initialize_global_variables(sess)
tl.files.exists_or_mkdir(save_dir)
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')

iter_counter = 0
max_epoch = 10000
mu = -3.5
std = 0.5
g_train_cnt = 2
d_train_cnt = 1

g_means = []
g_vars = []

for epoch in range(max_epoch):
    sample_fake = np.sort(np.random.uniform(size=(batch_size, sample_cnt)).astype(np.float), axis=1)
    sample_true = np.sort(np.random.normal(loc=mu, scale=std, size=(batch_size, sample_cnt)).astype(np.float), axis=1)

    start_time = time.time()
    for _ in range(d_train_cnt):
        errD, _, tm, tv = sess.run([d_loss, d_optim, t_mean, t_var], feed_dict={z: sample_fake, true_data: sample_true })
    for _ in range(g_train_cnt):
        errG, _, gm, gv = sess.run([g_loss, g_optim, g_mean, g_var], feed_dict={z: sample_fake})
    print("Epoch: [%2d/%2d] time: %4.4f, d_loss: %.8f, g_loss: %.8f, g_mean: %.8f, g_var: %.8f, t_mean: %.8f, t_var: %.8f"\
        % (epoch, max_epoch, time.time() - start_time, errD, errG, gm, gv, tm, tv))

    if epoch == max_epoch/2:
        sess.run(tf.assign(lr, lr/2))
    if epoch % 100 == 0:
        print('save train data')
        g_means.append(gm)
        g_vars.append(gv)

sess.close()

f = open('g_mean_var.txt','w')
for g_m in g_means:
    f.write(str(g_m))
    f.write('\t')
f.write('\n')
for g_v in g_vars:
    f.write(str(g_v))
    f.write('\t')
f.write('\n')
f.close()