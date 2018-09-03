import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# read the data
# df = pd.read_csv('data/mcelroy_dataclean.csv') # read data set using pandas
# df = pd.read_csv('data/wilkerson_dataclean.csv') # read data set using pandas
# df = pd.read_csv('data/combined.csv') # read data set using pandas
# df = pd.read_csv('data/combined_modified.csv') # read data set using pandas
df = pd.read_csv('data/combined_modified_cut.csv') # read data set using pandas
df = df.dropna(inplace=False)  # Remove all nan entries.
print('Data summary:\n')
print(df.describe(), '\n\n') # Overview of dataset

# subset for train and test and rescale all values
df_train, df_test = train_test_split(df, test_size=0.20)

scaler = MinMaxScaler() # For normalizing dataset

# we want to predict the H and B given Qbf, S, D50
# y is output and x is features

X_train = scaler.fit_transform(df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
y_train = scaler.fit_transform(df_train[['Bbf.m', 'Hbf.m']].values)

X_test = scaler.fit_transform(df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
y_test = scaler.fit_transform(df_test[['Bbf.m', 'Hbf.m']].values)


def denormalize(df, norm_data):
    """
    Above written function for denormalization of data after normalizing
    this function will give original scale of values.
    """

    df = df[['Bbf.m', 'Hbf.m']].values
    norm_data = norm_data
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
    return new


def neural_net_model(X_data, input_dim):
    """
    neural_net_model is function applying 2 hidden layer feed forward neural net.
    Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
    These are variables with will be updated during training.
    """
    n_nodes = 3

    # layer 1 multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim, n_nodes]))
    b_1 = tf.Variable(tf.zeros([n_nodes]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 2 multiplying and adding bias then activation function    
    # W_2 = tf.Variable(tf.random_uniform([n_nodes, n_nodes]))
    # b_2 = tf.Variable(tf.zeros([n_nodes]))
    # layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    # layer_2 = tf.nn.relu(layer_2)

    # output layer multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([n_nodes, 2])) # 2 because there are two outputs
    b_O = tf.Variable(tf.zeros([2]))
    output = tf.add(tf.matmul(layer_1, W_O), b_O)

    return output, W_O


xs = tf.placeholder("float", [None, X_train.shape[1]], name='x')
ys = tf.placeholder("float", [None, y_train.shape[1]], name='y')

# the model
output, W_O = neural_net_model(xs, X_train.shape[1])

# our mean squared error cost function
loss = tf.reduce_sum(tf.square(output - ys))

# Gradinent Descent optimiztion just discussed above for updating weights and biases
train = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# some other initializations
correct_pred = tf.argmax(output, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

c_train = []
c_test = []

with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())

    writer = tf.summary.FileWriter("logs/graph", sess.graph)

    saver = tf.train.Saver()
    #saver.restore(sess,'channel_geom_nn.ckpt')
    
    for i in range(500):
        for j in range(X_train.shape[0]):
            # Run loss and train with each sample
            sess.run([loss, train], feed_dict = {xs:X_train[j,:].reshape(1, X_train.shape[1]), 
                                                 ys:y_train[j,:].reshape(1, y_train.shape[1])})

        # keep track of the loss
        c_train.append(sess.run(loss, feed_dict = {xs:X_train, ys:y_train}))
        c_test.append(sess.run(loss, feed_dict = {xs:X_test, ys:y_test}))
        
        print('Epoch:', i, ', train loss:', c_train[i], ', test loss:', c_test[i])
    
    # if input('Save model ? [Y/N]') == 'Y':
    #     saver.save(sess,'channel_geom_nn.ckpt')
    #     print('Model Saved')

    # finished training
    print('Training complete.')
    writer.close()

    # predict output of test data after training
    pred = sess.run(output, feed_dict={xs:X_test})
    
    # denormalize data  
    print('test loss :', sess.run(loss, feed_dict={xs:X_test, ys:y_test}))
    y_test = denormalize(df_test, y_test)
    pred = denormalize(df_test, pred)

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
    ax2[0].plot(y_test[:,1], pred[:,1], 'o')
    ax2[0].plot([df['Hbf.m'].min(), df['Hbf.m'].max()], [df['Hbf.m'].min(), df['Hbf.m'].max()], 'k-', lw=2)
    ax2[0].axis('square')
    ax2[0].set_title('depth H, (m)')
    ax2[0].set_yscale('log')
    ax2[0].set_xscale('log')
    ax2[0].set_xlabel('actual')
    ax2[0].set_ylabel('pred')
    ax2[0].set_xlim([df['Hbf.m'].min(), df['Hbf.m'].max()])
    ax2[0].set_ylim([df['Hbf.m'].min(), df['Hbf.m'].max()])
    ax2[1].plot(y_test[:,0], pred[:,0], 'o')
    ax2[1].plot([df['Bbf.m'].min(), df['Bbf.m'].max()], [df['Bbf.m'].min(), df['Bbf.m'].max()], 'k-', lw=2)
    ax2[1].axis('square')
    ax2[1].set_title('width B, (m)')
    ax2[1].set_yscale('log')
    ax2[1].set_xscale('log')
    ax2[1].set_xlabel('actual')
    ax2[1].set_ylabel('pred')
    ax2[1].set_xlim([df['Bbf.m'].min(), df['Bbf.m'].max()])
    ax2[1].set_ylim([df['Bbf.m'].min(), df['Bbf.m'].max()])
    fig2.savefig('figures/compare.png')

    fig3, ax3 = plt.subplots()
    ax3.plot(np.arange(len(c_train)), np.array(c_train))
    ax3.plot(np.arange(len(c_test)), np.array(c_test))
    ax3.set_xlabel('epoch')
    ax3.set_ylabel('loss')
    plt.legend(['train', 'test'], loc = 'best')
    fig3.savefig('figures/train.png')


    