import tensorflow as tf
import tensorflow.contrib.layers as layers
class Model:

    #TODO : replace model to CNN

    def __init__(self,labels=None,input_shape = None):
        self.labels =labels
        self.input_dims = input_shape

    def encoder(self,X,name='encoder',sess=None):
        print('encoder')

        with tf.variable_scope(name):
            #encoder_x = tf.reshape(X,[1,-1])

            encoder_x = tf.layers.flatten(X)
            encoder_w = tf.get_variable(
                "w",
                shape=[encoder_x.shape[1], self.labels * 2],
                dtype=tf.float32,
                initializer= layers.xavier_initializer(),
            )
            encoder_b = tf.Variable(tf.random_normal([self.labels * 2]))
            affined_encoder = tf.matmul(encoder_x,encoder_w) + encoder_b
            self.lay_out = tf.nn.relu(affined_encoder)
            mean = self.lay_out[:,:self.labels]
            std_dev = 1e-6 + tf.nn.softplus(self.lay_out[:,self.labels:])

        return mean,std_dev


    def decoder(self,Z,name='decoder',sess=None):
        with tf.variable_scope(name):
            self.Z = Z
            decoder_w = tf.get_variable(
                "decoder_W",
                shape=[self.labels,self.input_dims[1]],
                dtype=tf.float32,
                initializer=layers.xavier_initializer()
            )
            decoder_b = tf.get_variable(
                "decoder_b",
                shape=[self.input_dims[1]],
                dtype=tf.float32,
                initializer=layers.xavier_initializer()
            )
            self.affined_decoder = tf.matmul(Z, decoder_w) + decoder_b
            self.out = tf.sigmoid(self.affined_decoder)
            #self.out = tf.clip_by_value(self.out,clip_value_min=1e-8,clip_value_max=1-(1e-8))
        return self.out

    def predict_decoder(self,Z,sess):
        return sess.run(self.out,feed_dict = {self.Z:Z})
