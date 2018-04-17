import tensorflow as tf
import tensorflow.contrib.layers as layers
class Model:
    def __init__(self,labels=None,input_shape = None):
        self.labels =labels
        self.input_dims = input_shape


    def encoder(self,X,name='encoder',sess=None):
        print('encoder')

        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
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
            std_dev = tf.nn.softplus(self.lay_out[:,self.labels:])

        return mean,std_dev


    def decoder(self,Z,name='decoder',sess=None):
        with tf.variable_scope(name,reuse=tf.AUTO_REUSE):
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
            #out = tf.tanh(affined_decoder)
        #return out
        return self.affined_decoder

    def predict_decoder(self,Z,sess):
        return sess.run(self.affined_decoder,feed_dict = {self.Z:Z})