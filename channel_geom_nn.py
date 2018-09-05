import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# read the data
df = pd.read_csv('data/mcelroy_dataclean.csv') # read data set using pandas
# df = pd.read_csv('data/wilkerson_dataclean.csv') # read data set using pandas
# df = pd.read_csv('data/combined.csv') # read data set using pandas
# df = pd.read_csv('data/combined_modified.csv') # read data set using pandas
# df = pd.read_csv('data/combined_modified_cut.csv') # read data set using pandas
df = df.dropna(inplace=False)  # Remove all nan entries.
print('Data summary:\n')
print(df.describe(), '\n\n') # Overview of dataset

# subset for train and test and rescale all values
df_train, df_test = train_test_split(df, test_size=0.30)

scaler = MinMaxScaler() # For normalizing dataset

# we want to predict the H and B given Qbf, S, D50
# y is output and x is features

# min max normalization
# X_train = scaler.fit_transform(df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
# y_train = scaler.fit_transform(df_train[['Bbf.m', 'Hbf.m']].values)
# X_test = scaler.fit_transform(df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
# y_test = scaler.fit_transform(df_test[['Bbf.m', 'Hbf.m']].values)
# logged = False
# normed = True

# min max log(x) normalization
# X_train = scaler.fit_transform(np.log10(df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
# y_train = scaler.fit_transform(np.log10(df_train[['Bbf.m', 'Hbf.m']].values))
# X_test = scaler.fit_transform(np.log10(df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
# y_test = scaler.fit_transform(np.log10(df_test[['Bbf.m', 'Hbf.m']].values))
# logged = True
# normed = True

# log(x) normalization
X_train = (np.log10(df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
y_train = (np.log10(df_train[['Bbf.m', 'Hbf.m']].values))
X_test = (np.log10(df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
y_test = (np.log10(df_test[['Bbf.m', 'Hbf.m']].values))
logged = True
normed = False

# min max log(x) normalization (whole dataset)
# X_train = scaler.fit_transform(np.log10(df.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
# y_train = scaler.fit_transform(np.log10(df[['Bbf.m', 'Hbf.m']].values))
# X_test = scaler.fit_transform(np.log10(df.drop(['Bbf.m', 'Hbf.m'], axis=1).values))
# y_test = scaler.fit_transform(np.log10(df[['Bbf.m', 'Hbf.m']].values))
# logged = True
# normed = True

# no normalization (be sure to turn off below for plotting)
# X_train = (df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
# y_train = (df_train[['Bbf.m', 'Hbf.m']].values)
# X_test = (df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
# y_test = (df_test[['Bbf.m', 'Hbf.m']].values)
# logged = False
# normed = False

batch_size = 1
ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).repeat().batch(batch_size)
it_train = ds_train.make_one_shot_iterator()
xs, ys = it_train.get_next()


def denormalize(df, norm_data):
    """
    Above written function for denormalization of data after normalizing
    this function will give original scale of values.
    """

    if logged:
        df = np.log10(df[['Bbf.m', 'Hbf.m']].values)
    else:
        df = df[['Bbf.m', 'Hbf.m']].values
    
    if normed:
        scl = MinMaxScaler()
        a = scl.fit_transform(df)
        new = scl.inverse_transform(norm_data)
    else:
        new = norm_data
    
    if logged:
        expt = np.exp(new)
        return expt
    else:
        return new


def neural_net_model(X_data, input_dim):
    """
    neural_net_model is function applying 2 hidden layer feed forward neural net.
    Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
    These are variables with will be updated during training.
    """
    
    n_nodes = 2

    # layer 1 multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim, n_nodes], dtype='float64'))
    b_1 = tf.Variable(tf.zeros([n_nodes], dtype = 'float64'))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)
    # layer_1 = tf.nn.tanh(layer_1)
    # layer_1 = tf.nn.sigmoid(layer_1)
    # layer_1 = tf.nn.softmax(layer_1)
    # layer_1 = tf.nn.leaky_relu(layer_1, alpha = 0.1)

    # layer 2 multiplying and adding bias then activation function    
    # W_2 = tf.Variable(tf.random_uniform([n_nodes, n_nodes], dtype='float64'))
    # b_2 = tf.Variable(tf.zeros([n_nodes], dtype = 'float64'))
    # layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    # layer_2 = tf.nn.relu(layer_2)

    # output layer multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([n_nodes, 2], dtype = 'float64')) # 2 because there are two outputs
    b_O = tf.Variable(tf.zeros([2], dtype = 'float64'))
    output = tf.add(tf.matmul(layer_1, W_O), b_O)
    # output = tf.pow(tf.matmul(layer_1, W_O), b_O)

    return output, W_O


# xs = tf.placeholder("float", [None, X_train.shape[1]], name='x')
# ys = tf.placeholder("float", [None, y_train.shape[1]], name='y')

# the model
output, W_O = neural_net_model(xs, X_train.shape[1])

# mean squared error cost function
loss = tf.reduce_sum(tf.square(output - ys))
# loss = tf.reduce_mean(tf.square(output - ys))
# loss = tf.losses.mean_squared_error(output, ys)

# Gradinent Descent optimiztion just discussed above for updating weights and biases
learning_rate = 0.01
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
# train = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# some other initializations
# correct_pred = tf.argmax(output, 1)
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

c_train = []
c_test = []

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("logs/graph", sess.graph)

    saver = tf.train.Saver()
    #saver.restore(sess,'channel_geom_nn.ckpt')

    # batcher = tf.train.batch(, batch_size=10, allow_smaller_final_batch = True)

    it = 0
    n_epoch = 10
    n_batch_per_epoch = int( np.floor(X_train.shape[0] / batch_size) )
    for i in range(n_epoch):
        for j in range(n_batch_per_epoch):
            # Run loss and train with each sample (1 sample per batch)
            # sess.run([loss, train], feed_dict = {xs:X_train[j,:].reshape(1, X_train.shape[1]), 
            #                                      ys:y_train[j,:].reshape(1, y_train.shape[1])})

            # Run loss and train with each sample (10 samples per batch)
            # batch_x, batch_y = batcher.next_batch(batch_size)
            sess.run([loss, train])

            # keep track of the loss
            c_train.append(sess.run(loss, feed_dict = {xs:X_train, ys:y_train}))
            c_test.append(sess.run(loss, feed_dict = {xs:X_test, ys:y_test}))
            it += 1
        
        print('Epoch:', i, ', train loss:', c_train[i*n_batch_per_epoch], ', test loss:', c_test[i*n_batch_per_epoch])
    
    # if input('Save model ? [Y/N]') == 'Y':
    #     saver.save(sess,'channel_geom_nn.ckpt')
    #     print('Model Saved')

    # finished training
    print('\nTraining complete.')
    print('Total iterations: ', it)
    print('test loss :', sess.run(loss, feed_dict={xs:X_test, ys:y_test}), '\n')
    writer.close()

    # predict output of test data after training
    pred_test = sess.run(output, feed_dict={xs:X_test})
    pred_train = sess.run(output, feed_dict={xs:X_train})
    
    # denormalize data
    y_test = denormalize(df_test, y_test)
    pred_test = denormalize(df_test, pred_test)
    y_train = denormalize(df_train, y_train)
    pred_train = denormalize(df_train, pred_train)

    fig1, axes1 = plt.subplots(nrows=1, ncols=3)
    axes1[0].hist([df_train['Qbf.m3s'], df_test['Qbf.m3s']], histtype = 'bar', density = True)
    axes1[0].set_xlabel('Qbf (m3/s)')
    axes1[1].hist([df_train['S'], df_test['S']], histtype = 'bar', density = True)
    axes1[1].set_xlabel('S')
    axes1[2].hist([df_train['D50.mm'], df_test['D50.mm']], histtype = 'bar', density = True)
    axes1[2].set_xlabel('D50 (mm)')
    plt.legend(['train', 'test'], loc = 'best')
    fig1.savefig('figures/split.png')

    fig2, ax2 = plt.subplots(nrows=1, ncols=2)
    ax2[0].plot(df_train['Hbf.m'], pred_train[:,1], 'o', alpha=0.2)
    ax2[0].plot(df_test['Hbf.m'], pred_test[:,1], 'o', alpha=0.2)
    ax2[0].plot([df['Hbf.m'].min()/10, df['Hbf.m'].max()*10], [df['Hbf.m'].min()/10, df['Hbf.m'].max()*10], 'k-', lw=2)
    ax2[0].axis('square')
    ax2[0].set_title('depth H, (m)')
    ax2[0].set_yscale('log')
    ax2[0].set_xscale('log')
    ax2[0].set_xlabel('actual')
    ax2[0].set_ylabel('pred')
    ax2[0].set_xlim([df['Hbf.m'].min()/10, df['Hbf.m'].max()*10])
    ax2[0].set_ylim([df['Hbf.m'].min()/10, df['Hbf.m'].max()*10])
    ax2[1].plot(df_train['Bbf.m'], pred_train[:,0], 'o', alpha=0.2)
    ax2[1].plot(df_test['Bbf.m'], pred_test[:,0], 'o', alpha=0.2)
    ax2[1].plot([df['Bbf.m'].min()/10, df['Bbf.m'].max()*10], [df['Bbf.m'].min()/10, df['Bbf.m'].max()*10], 'k-', lw=2)
    ax2[1].axis('square')
    ax2[1].set_title('width B, (m)')
    ax2[1].set_yscale('log')
    ax2[1].set_xscale('log')
    ax2[1].set_xlabel('actual')
    ax2[1].set_ylabel('pred')
    ax2[1].set_xlim([df['Bbf.m'].min()/10, df['Bbf.m'].max()*10])
    ax2[1].set_ylim([df['Bbf.m'].min()/10, df['Bbf.m'].max()*10])
    fig2.savefig('figures/compare.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(np.arange(len(c_train)) / n_batch_per_epoch, np.array(c_train))
    ax3.plot(np.arange(len(c_test)) / n_batch_per_epoch, np.array(c_test))
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('loss')
    plt.legend(['train', 'test'], loc = 'best')
    fig3.savefig('figures/train.png')

    # print(W_O.read_value().eval())
    