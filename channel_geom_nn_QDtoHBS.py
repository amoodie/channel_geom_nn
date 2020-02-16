import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import abc

import dataset
import plot_utils


tf.reset_default_graph()


data = dataset.ChannelDataset(datasrc='trampush', scale='log')



class BaseModel(object):
    def __init__(self, input_tensor, loss_flag, input_dim, output_dim,
                 learning_rate, optimizer_flag):
        self.loss_flag = loss_flag
        self.input_tensor = input_tensor
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.optimizer_flag = optimizer_flag
        self.learning_rate = learning_rate


    @abc.abstractmethod
    def set_arch(self):
        pass


    def set_loss(self, ys):
        if self.loss_flag == 'mse':
            self.loss = tf.losses.mean_squared_error(self.output_tensor, ys)
        else:
            raise ValueError('Bad loss_flag given: %s' % self.loss_flag)
        

    def set_optimizer(self):
        if self.optimizer_flag == 'GD':
            self.optimizer = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss)
        elif self.optimizer_flag == 'Adam':
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        else:
            raise ValueError('Bad optimizer_flag given: %s' % self.optimizer_flag)



class NNModel(BaseModel):
    def __init__(self, input_tensor, loss_flag, input_dim, output_dim, 
                 n_layer=1, n_node=3, learning_rate=0.01, optimizer_flag='GD'):
        super().__init__(input_tensor, loss_flag, input_dim, output_dim, 
                         learning_rate, optimizer_flag)
        self.n_layer = n_layer
        self.n_node = n_node
        assert n_layer <= 2, 'No more than 2 hidden layers'

        self.output_tensor, self.W_0 = self.set_arch()


    def set_arch(self):
        """
        Constructs the neural network model. 
        It can be a 1 layer or 2 layer model, with n_nodes.
        """
        indim = self.input_dim
        inten = self.input_tensor
        W = []
        b = []

        for i in range(self.n_layer):
            W.append(tf.Variable(tf.random_uniform([indim, self.n_node], dtype='float64')))
            b.append(tf.Variable(tf.zeros([self.n_node], dtype = 'float64')))
            op = tf.add(tf.matmul(inten, W[i]), b[i])
            op = tf.nn.relu(op)
            indim = self.n_node
            inten = op

        W_O = tf.Variable(tf.random_uniform([self.n_node, self.output_dim], dtype = 'float64')) # 3 because there are two outputs
        b_O = tf.Variable(tf.zeros([self.output_dim], dtype = 'float64'))
        output = tf.add(tf.matmul(op, W_O), b_O)

        return output, W_O


    
# get an initial dataset
xs, ys = data.get_next()
input_dim = data.X_train.shape[1]
output_dim = data.y_train.shape[1]


model_flag = 'nnmodel'
if model_flag == 'nnmodel':
    model = NNModel(input_tensor=xs, loss_flag='mse',
                    input_dim=input_dim, output_dim=output_dim,
                    n_layer=2, n_node=2)
else:
    raise ValueError('Bad model_flag supplied: %s' % model_flag)


model.set_loss(ys)
model.set_optimizer()

# some other initializations
_loss_summary = tf.summary.scalar(name='loss summary', tensor=model.loss)
saver = tf.train.Saver()

c_train = []
c_test = []

save_training = False # whether to save images every iteration for gif

with tf.Session() as sess: # initiate session
    
    sess.run(tf.global_variables_initializer()) # initialize all vaiables

    writer = tf.summary.FileWriter("log/", sess.graph)

    it = 0
    n_epoch = 10
    
    for i in range(n_epoch):
        
        data.shuffle()
        
        for j in range(data.batches_per_epoch):
            # Run loss and train with each batch
            sess.run([model.loss, model.optimizer])

            c_train.append(sess.run(model.loss, feed_dict = {xs:data.X_train, ys:data.y_train}))
            c_test.append(sess.run(model.loss, feed_dict = {xs:data.X_test, ys:data.y_test}))

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
    print('test loss :', sess.run(model.loss, feed_dict={xs:data.X_test, ys:data.y_test}), '\n')
    writer.close()

    # save the model
    save_path = saver.save(sess, "log/channel_geom_nn_QDtoHBS.ckpt")

    # predict output of test data after training
    pred_test = sess.run(model.output_tensor, feed_dict={xs:data.X_test})
    pred_train = sess.run(model.output_tensor, feed_dict={xs:data.X_train})
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
ax5.matshow(data.df.corr())
fig5.savefig('figures/corr_mat.png')
