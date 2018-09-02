import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf

# read the data
df = pd.read_csv('mcelroy_dataclean.csv') # read data set using pandas
df = df.dropna(inplace=False)  # Remove all nan entries.
print(df.describe()) # Overview of dataset

# subset for train and test and rescale all values
df_train, df_test = train_test_split(df, test_size=0.15)

scaler = MinMaxScaler() # For normalizing dataset

# we want to predict the H and B given Qbf, S, D50
# y is output and x is features

X_train = scaler.fit_transform(df_train.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
y_train = scaler.fit_transform(df_train[['Bbf.m', 'Hbf.m']].values)

X_test = scaler.fit_transform(df_test.drop(['Bbf.m', 'Hbf.m'], axis=1).values)
y_test = scaler.fit_transform(df_test[['Bbf.m', 'Hbf.m']].values)

def denormalize(df, norm_data):
    df = df['Close'].values.reshape(-1,1)
    norm_data = norm_data.reshape(-1,1)
    scl = MinMaxScaler()
    a = scl.fit_transform(df)
    new = scl.inverse_transform(norm_data)
"""
Above written function for denormalizatio of data after normalizing
this function will give original scale of values.
In normalization we step down the value of data in dataset.
"""

def neural_net_model(X_data, input_dim):
    """
    neural_net_model is function applying 2 hidden layer feed forward neural net.
    Weights and biases are abberviated as W_1,W_2 and b_1, b_2 
    These are variables with will be updated during training.
    """

    # layer 1 multiplying and adding bias then activation function
    W_1 = tf.Variable(tf.random_uniform([input_dim, 10]))
    b_1 = tf.Variable(tf.zeros([10]))
    layer_1 = tf.add(tf.matmul(X_data, W_1), b_1)
    layer_1 = tf.nn.relu(layer_1)

    # layer 2 multiplying and adding bias then activation function    
    W_2 = tf.Variable(tf.random_uniform([10, 10]))
    b_2 = tf.Variable(tf.zeros([10]))
    layer_2 = tf.add(tf.matmul(layer_1, W_2), b_2)
    layer_2 = tf.nn.relu(layer_2)

    # output layer multiplying and adding bias then activation function
    W_O = tf.Variable(tf.random_uniform([10,2]))
    b_O = tf.Variable(tf.zeros([2]))
    output = tf.add(tf.matmul(layer_2, W_O), b_O)

    return output


xs = tf.placeholder("float", [None, X_train.shape[1]], name='x')
ys = tf.placeholder("float", [None, y_train.shape[1]], name='y')

# the model
output = neural_net_model(xs, X_train.shape[1])

# our mean squared error cost function
cost = tf.reduce_mean(tf.square(output-ys))

# Gradinent Descent optimiztion just discussed above for updating weights and biases
train = tf.train.GradientDescentOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
    # Initiate session and initialize all vaiables
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    #saver.restore(sess,'yahoo_dataset.ckpt')
    for i in range(10):
        for j in range(X_train.shape[0]):
            # Run cost and train with each sample
            sess.run([cost, train], feed_dict = {xs:X_train[j,:].reshape(X_train.shape[1], 1), ys:y_train[j,:]})
        
        c_t.append(sess.run(cost, feed_dict = {xs:X_train, ys:y_train}))
        c_test.append(sess.run(cost, feed_dict = {xs:X_test, ys:y_test}))
        
        print('Epoch :',i,'Cost :',c_t[i])
    
    # finished training
    print('Training complete.')

    # predict output of test data after training
    pred = sess.run(output, feed_dict={xs:X_test})
    
    # print('Cost :',sess.run(cost, feed_dict={xs:X_test,ys:y_test}))
    # y_test = denormalize(df_test,y_test)
    # pred = denormalize(df_test,pred)
    # #Denormalize data     

    # plt.plot(range(y_test.shape[0]),y_test,label="Original Data")
    # plt.plot(range(y_test.shape[0]),pred,label="Predicted Data")
    # plt.legend(loc='best')
    # plt.ylabel('Stock Value')
    # plt.xlabel('Days')
    # plt.title('Stock Market Nifty')
    # plt.show()
    # if input('Save model ? [Y/N]') == 'Y':
    #     saver.save(sess,'yahoo_dataset.ckpt')
    #     print('Model Saved')