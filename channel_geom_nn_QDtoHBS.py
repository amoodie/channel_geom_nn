import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

import dataset
import plot_utils


tf.reset_default_graph()


data = dataset.ChannelDataset(datasrc='trampush', scale='log')


def denormalize(df, norm_data):
    """
    Above written function for denormalization of data after normalizing
    this function will give original scale of values.
    """

    if logged:
        df = np.log10(df[['Bbf.m', 'Hbf.m', 'S']].values)
    else:
        df = df[['Bbf.m', 'Hbf.m', 'S']].values

    if normed:
        scl = MinMaxScaler()
        a = scl.fit_transform(df)
        new = scl.inverse_transform(norm_data)
    else:
        new = norm_data
    
    if logged:
        expt = np.power(10, new)
        return expt
    else:
        return new


def nn_model(X_data, input_dim):
    """
    nn_model constructs the neural network model. 
    It can be a 1 layer or 2 layer model, with n_nodes.
    Weights and biases are abberviated as W_1, W_2 and b_1, b_2 
    """

    n_nodes = 1

    # layer 1 multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim, n_nodes], dtype='float64'))
    b_1 = tf.Variable(tf.zeros([n_nodes], dtype = 'float64'))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 2 multiplying and adding bias then activation function    
    # W_2 = tf.Variable(tf.random_uniform([n_nodes, n_nodes], dtype='float64'))
    # b_2 = tf.Variable(tf.zeros([n_nodes], dtype = 'float64'))
    # layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    # layer_2 = tf.nn.relu(layer_2)

    # output layer multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([n_nodes, 3], dtype = 'float64')) # 3 because there are two outputs
    b_O = tf.Variable(tf.zeros([3], dtype = 'float64'))
    output = tf.add(tf.matmul(layer_1, W_O), b_O)
    # output = tf.add(tf.matmul(layer_2, W_O), b_O)

    return output, W_O

# get an initial dataset
xs, ys = data.get_next()

# the model
output, W_O = nn_model(xs, data.X_train.shape[1])

# mean squared error cost function
# loss = tf.reduce_sum(tf.square(output - ys))
# loss = tf.reduce_mean(tf.square(output - ys))
loss = tf.losses.mean_squared_error(output, ys)

# Gradinent Descent optimiztion just discussed above for updating weights and biases
learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# some other initializations
_loss_summary = tf.summary.scalar(name='loss summary', tensor=loss)
# correct_pred = tf.argmax(output, 1)
# accuracy = tf.losses.mean_squared_error(tf.cast(correct_pred, tf.float32), ys)
saver = tf.train.Saver()

c_train = []
c_test = []

save_training = True
# intrain_y_test = data.y_scaler.inverse_transform(y_test)
# intrain_y_train = data.y_scaler.inverse_transform(data.y_train)

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("log/", sess.graph)

    it = 0
    n_epoch = 10
    
    for i in range(n_epoch):
        
        data.shuffle()
        
        for j in range(data.batches_per_epoch):
            # Run loss and train with each batch
            sess.run([loss, train])

            c_train.append(sess.run(loss, feed_dict = {xs:data.X_train, ys:data.y_train}))
            c_test.append(sess.run(loss, feed_dict = {xs:data.X_test, ys:data.y_test}))

            loss_summary = sess.run(_loss_summary)
            writer.add_summary(loss_summary, it)

            if save_training:
                
                pred_test = sess.run(output, feed_dict={xs:data.X_test})
                pred_train = sess.run(output, feed_dict={xs:data.X_train})
                pred_test_orig = data.y_scaler.inverse_transform(pred_test)
                pred_train_orig = data.y_scaler.inverse_transform(pred_train)

                figN = plot_utils.compare_plot(data.df, data.y_train_orig, data.y_test_orig, 
                                               pred_train_orig, pred_test_orig)
                figN.savefig('figures/training/{:04d}.png'.format(it))
                plt.close(figN)
                

            it += 1
        
        print('Epoch:', i, ', train loss:', c_train[-1], ', test loss:', c_test[-1])

    # finished training
    print('\nTraining complete.')
    print('Total iterations: ', it)
    print('test loss :', sess.run(loss, feed_dict={xs:data.X_test, ys:data.y_test}), '\n')
    writer.close()

    # save the model
    save_path = saver.save(sess, "log/channel_geom_nn_QDtoHBS.ckpt")

    # predict output of test data after training
    pred_test = sess.run(output, feed_dict={xs:data.X_test})
    pred_train = sess.run(output, feed_dict={xs:data.X_train})
    pred_test_orig = data.y_scaler.inverse_transform(pred_test)
    pred_train_orig = data.y_scaler.inverse_transform(pred_train)

# plots
fig1, axes1 = plt.subplots(nrows=1, ncols=2, figsize=(6,4))
axes1[0].hist([data.df_train['Qbf.m3s'], data.df_test['Qbf.m3s']], histtype = 'bar', density = True)
axes1[0].set_xlabel('Qbf (m3/s)')
axes1[1].hist([data.df_train['D50.mm'], data.df_test['D50.mm']], histtype = 'bar', density = True)
axes1[1].set_xlabel('D50 (mm)')
plt.legend(['train', 'test'], loc = 'best')
fig1.savefig('figures/split.png')

fig2 = plot_utils.compare_plot(data.df,  data.y_train_orig, data.y_test_orig, 
                               pred_train_orig, pred_test_orig)
fig2.savefig('figures/compare.png')

fig3, ax3 = plt.subplots(figsize=(6,4))
ax3.plot(np.arange(len(c_train)) / data.batches_per_epoch, np.array(c_train))
ax3.plot(np.arange(len(c_test)) / data.batches_per_epoch, np.array(c_test))
ax3.set_xlabel('epoch')
ax3.set_ylabel('loss')
plt.legend(['train', 'test'], loc = 'best')
fig3.savefig('figures/train.png')

fig4, ax4 = plt.subplots(figsize=(8,6))
pd.plotting.scatter_matrix(np.log10(data.df), ax=ax4)
fig4.savefig('figures/scatter.png')

fig5, ax5 = plt.subplots()
ax5.matshow(np.log10(data.df.corr()))
fig5.savefig('figures/corr_mat.png')
