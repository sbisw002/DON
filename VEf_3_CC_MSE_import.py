#!/usr/bin/env python3


from __future__ import absolute_import, division, print_function

#import tensorflow as tf
import numpy as np
import scipy.io as sio
import os
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

cwd = os.getcwd()
os.chdir(cwd)

mat = sio.loadmat('data_4_training_Ef_1000.mat')    #loading data sets with prob >= 0.99 and not noisy
V = mat['V_train']
Ef = mat['Ef_train']
Ef_preprocess = 'multiply_by_80'
Ef = Ef * 80
No = 36118
No_train = 26000
No_val = 5000
No_test = No - No_train

t_min = 20.0
t_max = 100.0
tau = 0.1
t_4_Ef = np.arange(t_min + tau/2, t_max + tau/2, tau)

learning_rate = 0.0001
training_epoch = 40000
batch_size = 1024      #must be larger than 1
display_step = 1000
shuffle_size = No_train
prefetch_size = 1
#training parameters

V_train = V[:No_train]
V_val = V[No_train:(No_train + No_val)]
V_test = V[(No_train + No_val):]    #x, input
Ef_train = Ef[:No_train]
Ef_val = Ef[No_train:(No_train + No_val)]
Ef_test = Ef[(No_train + No_val):]    #y, output

loss_training = []
rel_pos_training = []
rel_amp_training = []
loss_val = []
rel_pos_val = []
rel_amp_val = []

V_train, V_val, V_test = np.array(V_train, np.float32), np.array(V_val, np.float32), np.array(V_test, np.float32)
Ef_train, Ef_val, Ef_test = np.array(Ef_train, np.float32), np.array(Ef_val, np.float32), np.array(Ef_test, np.float32)

# Use tf.data API to shuffle and batch data:
train_data = tf.data.Dataset.from_tensor_slices((V_train, Ef_train))
train_data = train_data.repeat().shuffle(shuffle_size).batch(batch_size).prefetch(prefetch_size)

#neural network size:
num_V = 192    #potential in 192 points
num_h1 = 300
num_h2 = 450
num_h3 = 600
num_pred = 800
 
num_cc_maxshift = 150     #the maximum shift of cross-correlation, should be smaller than half of num_pred

def gen_shift_matrix(num_cc_maxshift, num_pred):
    shift_matrix = np.zeros((num_cc_maxshift, num_pred, num_pred), np.float32)
    for shift_i in range(num_cc_maxshift):
        shift_matrix[shift_i] = np.roll(np.identity(num_pred, np.float32), shift = shift_i, axis = 1)
    shift_matrix = tf.cast(shift_matrix, tf.float32)
    return shift_matrix
    
shift_matrix = gen_shift_matrix(num_cc_maxshift, num_pred)   #make sure num_cc_maxshift < num_pred

def gen_idn_matrix(num_cc_maxshift, num_pred):
    idn_matrix = np.zeros((num_cc_maxshift, num_pred, num_pred), np.float32)
    for idn_i in range(num_cc_maxshift):
        idn_matrix[idn_i] = np.identity(num_pred, np.float32)
    idn_matrix = tf.cast(idn_matrix, tf.float32)
    return idn_matrix
    
idn_matrix = gen_idn_matrix(num_cc_maxshift, num_pred)   #make sure num_cc_maxshift < num_pred
    
# A random value generator to initialize weights.
random_normal = tf.initializers.RandomNormal()

# Parameters of weights and biases:
W1 = tf.Variable(random_normal([num_V, num_h1]) , dtype = tf.float32)
b1 = tf.Variable(0.001 * tf.ones([num_h1]) , dtype = tf.float32)
W2 = tf.Variable(random_normal([num_h1, num_h2]) , dtype = tf.float32)
b2 = tf.Variable(0.001 * tf.ones([num_h2]) , dtype = tf.float32)
W3 = tf.Variable(random_normal([num_h2, num_h3]) , dtype = tf.float32)
b3 = tf.Variable(0.001 * tf.ones([num_h3]) , dtype = tf.float32)
W4 = tf.Variable(random_normal([num_h3, num_pred]) , dtype = tf.float32)
b4 = tf.Variable(0.001 * tf.ones([num_pred]) , dtype = tf.float32)

# Use relu activation:
activation = 'relu_tanh'
def neural_network(x):
    a1 = x   # shape: [batch_size, num_V]
    a2 = tf.nn.relu(tf.add(tf.matmul(a1, W1) , b1))  # shape: [batch_size, num_h1]
    a3 = tf.nn.relu(tf.add(tf.matmul(a2, W2) , b2))  # shape: [batch_size, num_h2]
    a4 = tf.nn.relu(tf.add(tf.matmul(a3, W3) , b3))  # shape: [batch_size, num_h3]
    pred = tf.math.tanh(tf.add(tf.matmul(a4, W4) , b4))  # shape: [batch_size, num_pred]
    return pred

# Loss function:
loss_def = 'MSE_L2'
reg_coefficient = 0.001
def loss_func(y_pred, y_true):
    y_pred = tf.tensordot(y_pred, shift_matrix, axes = [[1], [1]]) # shape: [batch_size, num_cc_maxshift, num_pred]
    y_true = tf.tensordot(y_true, idn_matrix, axes = [[1], [1]]) # tf.tile & tf.broadcast_to don't work
    loss = tf.reduce_sum(tf.square(y_true - y_pred), axis = 2) # shape: [batch_size, num_cc_maxshift]
    loss = tf.math.reduce_min(loss, 1)
    loss = tf.reduce_mean(loss)
    
    regularizer = ( tf.reduce_mean(tf.nn.l2_loss(W1)) + tf.reduce_mean(tf.nn.l2_loss(W2)) 
                   + tf.reduce_mean(tf.nn.l2_loss(W3)) + tf.reduce_mean(tf.nn.l2_loss(W4)) )
    loss = loss + reg_coefficient * regularizer
    return loss

# Relativity metric:
def relativity(y_pred, y_true):
    y_pred = np.fft.fft(y_pred, n = num_pred)
    y_pred = np.absolute(y_pred[:, np.arange(int(num_pred/2))])
    y_true = np.fft.fft(y_true, n = num_pred)
    y_true = np.absolute(y_true[:, np.arange(int(num_pred/2))])
    
    valuemax_pred = np.amax(y_pred, axis = 1)
    argmax_pred = np.argmax(y_pred, axis = 1) + 1.0
    valuemax_true = np.amax(y_true, axis = 1)
    argmax_true = np.argmax(y_true, axis = 1) + 1.0
    
    valuemax_pred = tf.cast(valuemax_pred, tf.float32)
    argmax_pred = tf.cast(argmax_pred, tf.float32)
    valuemax_true = tf.cast(valuemax_true, tf.float32)
    argmax_true = tf.cast(argmax_true, tf.float32)
    
    diff_arg_square = tf.math.square(argmax_true - argmax_pred)
    true_arg_square = tf.math.square(argmax_true)
    relativity_pos = tf.reduce_mean(diff_arg_square / true_arg_square)
    relativity_pos = tf.cast(( 1.0 - relativity_pos ), tf.float32)
    
    diff_value_square = tf.math.square(valuemax_true - valuemax_pred)
    true_value_square = tf.math.square(valuemax_true)
    relativity_amp = tf.reduce_mean(diff_value_square / true_value_square)
    relativity_amp = tf.cast(( 1.0 - relativity_amp ), tf.float32)
    return relativity_pos, relativity_amp, valuemax_pred, argmax_pred, valuemax_true, argmax_true

optimizer_method = 'Adam'
# Adam optimizer:
optimizer = tf.optimizers.Adam(learning_rate)

# Optimization process:
def run_optimization(x, y):
    # Wrap computation inside a GradientTape for automatic differentiation:
    with tf.GradientTape() as g:
        pred = neural_network(x)
        loss = loss_func(pred, y)

    # Variables to update, i.e. trainable variables.
    trainable_variables = [W1, W2, W3, W4, b1, b2, b3, b4]

    # Compute gradients:
    gradients = g.gradient(loss, trainable_variables)    
    
    # Update W and b following gradients.
    optimizer.apply_gradients(zip(gradients, trainable_variables))

display_count = 0

# Run training for the given number of steps.
for step, (batch_x, batch_y) in enumerate(train_data.take(training_epoch), 1):
    # Run the optimization to update W and b values.
    #with tf.device('/device:cpu:0'):
    run_optimization(batch_x, batch_y)
    
    if step % display_step == 0:
        pred = neural_network(batch_x)
        loss = loss_func(pred, batch_y)
        rel_pos, rel_amp, _, _, _, _ = relativity(pred, batch_y)
        print("step: %i, training_loss: %f, training_pos_rel: %f, training_amp_rel: %f" % (step, loss, rel_pos, rel_amp))
        loss_training.append(loss)
        rel_pos_training.append(rel_pos)
        rel_amp_training.append(rel_amp)
        
        pred = neural_network(V_val)
        loss = loss_func(pred, Ef_val)
        rel_pos, rel_amp, _, _, _, _ = relativity(pred, Ef_val)
        print("step: %i, val_loss: %f, val_pos_rel: %f, val_amp_rel: %f" % (step, loss, rel_pos, rel_amp))
        loss_val.append(loss)
        rel_pos_val.append(rel_pos)
        rel_amp_val.append(rel_amp)
        
        display_count = display_count + 1

# Test model on validation set.
pred = neural_network(V_test)
loss = loss_func(pred, Ef_test)
rel_pos, rel_amp, valuemax_pred, argmax_pred, valuemax_true, argmax_true = relativity(pred, Ef_test)
print("Test loss: %f" % loss)
print("Test position relativity: %f" % rel_pos)
print("Test amplitude relativity: %f" % rel_amp)

loss_test = loss
rel_pos_test = rel_pos
rel_amp_test = rel_amp
valuemax_pred_test = valuemax_pred
argmax_pred_test = argmax_pred
valuemax_true_test = valuemax_true
argmax_true_test = argmax_true

def pred_shift(predictions, Ef_for_plt, n_plt, num_pred):
    predictions = tf.tensordot(predictions, shift_matrix, axes = [[1], [1]]) # shape: [n_plt, num_cc_maxshift, num_pred]
    Ef_for_plt = tf.tensordot(Ef_for_plt, idn_matrix, axes = [[1], [1]]) # shape: [n_plt, num_cc_maxshift, num_pred]
    loss = tf.reduce_sum(tf.square(Ef_for_plt - predictions), axis = 2) # shape: [n_plt, num_cc_maxshift]
    loss = loss.numpy()
    loss = np.argmin(loss, axis = 1)
    predictions = predictions.numpy()
    opt_pred = np.zeros((n_plt, num_pred), dtype=np.float)
    for plt_i in range(n_plt):
        opt_pred[plt_i] = predictions[plt_i][int(loss[plt_i])]
    return opt_pred

# Predict 30 images from validation set.
n_plt = 30
test_for_plt = V_test[:n_plt]
Ef_for_plt = Ef_test[:n_plt]

predictions = neural_network(test_for_plt)
predictions = pred_shift(predictions, Ef_for_plt, n_plt, num_pred)

'''
# Display image and model prediction.
for i in range(n_plt):
    plt.figure()
    plt.plot(t_4_Ef, Ef_for_plt[i], 'r')
    plt.plot(t_4_Ef, predictions[i], 'b')
    plt.xlabel('$time$')
    plt.ylabel('$Electric Field')
    plt.show()
    
plt.figure()
plt.plot(valuemax_true_test, valuemax_pred_test, 'or')
plt.xlabel('$amplitude_true$')
plt.ylabel('$prediction')
plt.show()

plt.figure()
plt.plot(argmax_true_test, argmax_pred_test, 'or')
plt.xlabel('$position_true$')
plt.ylabel('$prediction')
plt.show()

def scatterdensity(x, y, limit, text):
    x_y = np.vstack([x, y])
    z = gaussian_kde(x_y)(x_y)
    plt.scatter(x, y, c=z, s=30, edgecolors='')
    plt.xlabel('true')
    plt.ylabel(text)
    plt.xlim(0,limit)
    plt.ylim(0,limit)

plt.figure()
scatterdensity(valuemax_true_test, valuemax_pred_test, 1000, 'predicted amp')

plt.figure()
scatterdensity(argmax_true_test, argmax_pred_test, 20, 'predicted pos')

'''
 
curr_name = 'VEf_3_CC_MSE_new_2.mat'

W1, W2, W3, W4 = W1.numpy(), W2.numpy(), W3.numpy(), W4.numpy()
b1, b2, b3, b4 = b1.numpy(), b2.numpy(), b3.numpy(), b4.numpy()
loss_test, rel_pos_test, rel_amp_test = loss_test.numpy(), rel_pos_test.numpy(), rel_amp_test.numpy()
valuemax_pred_test, argmax_pred_test, valuemax_true_test, argmax_true_test = valuemax_pred_test.numpy(), argmax_pred_test.numpy(), valuemax_true_test.numpy(), argmax_true_test.numpy()

sio.savemat(curr_name, {'W1':W1,
                             'W2':W2,
                             'W3':W3,
                             'W4':W4,
                             'b1':b1,
                             'b2':b2,
                             'b3':b3,
                             'b4':b4,
                             'num_V':num_V, 
                             'num_h1':num_h1,
                             'num_h2':num_h2,
                             'num_h3':num_h3,
                             'num_pred':num_pred,
                             'num_cc_maxshift':num_cc_maxshift,
                             'Ef_preprocess':Ef_preprocess,
                             'test_loss': loss_test,
                             'test_position_relativity': rel_pos_test,
                             'test_amplitude_relativity': rel_amp_test,
                             'valuemax_pred_test': valuemax_pred_test,
                             'argmax_pred_test': argmax_pred_test,
                             'valuemax_true_test': valuemax_true_test,
                             'argmax_true_test': argmax_true_test,
                             'predictions': predictions,
                             'val_loss': loss_val,
                             'val_position_relativity': rel_pos_val,
                             'val_amplitude_relativity': rel_amp_val,
                             'training_loss': loss_training,
                             'training_position_relativity': rel_pos_training,
                             'training_amplitude_relativity': rel_amp_training,
                             'activation':activation,
                             'loss_def':loss_def,
                             'reg_coefficient': reg_coefficient,
                             'No_V':No,
                             'No_train':No_train,
                             'No_val':No_val,
                             'No_test':No_test,
                             'optimizer_method': optimizer_method,
                             'learning_rate':learning_rate,
                             'training_epoch':training_epoch,
                             'batch_size':batch_size,
                             'shuffle_size':shuffle_size,
                             'prefetch_size':prefetch_size})

