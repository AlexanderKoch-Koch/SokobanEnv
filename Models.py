import tensorflow as tf


class simple_DNN():

    def __init__(self, architecture, weight_mean=0.0):
        self.tf_x = tf.placeholder(shape=(architecture[0]), dtype=tf.float32, name="x")
        self.tf_y = tf.placeholder(shape=(architecture[-1]), dtype=tf.float32, name="y")

        """self.l0 = tf.transpose(tf.expand_dims(self.tf_x, axis=1))
        self.w1 = tf.Variable(tf.truncated_normal(shape=(input_length, architecture[0]), mean=weight_mean))
        self.b1 = tf.Variable(tf.truncated_normal(shape=(1, architecture[0]), mean=weight_mean))
        self.l1 = tf.nn.relu(tf.add(tf.matmul(self.l0, self.w1), self.b1))"""

        self.layer_outputs = []
        self.layer_weights = []
        self.layer_biases = []

        for layer in range(len(architecture) - 1):
            # input layer
            weights = tf.Variable(
                tf.truncated_normal(shape=(architecture[layer], architecture[layer + 1]), mean=weight_mean))
            biases = tf.Variable(tf.truncated_normal(shape=(1, architecture[layer + 1]), mean=weight_mean))
            if layer == 0:
                l0 = tf.transpose(tf.expand_dims(self.tf_x, axis=1))
                output = tf.nn.relu(tf.add(tf.matmul(l0, weights), biases))
            elif layer == len(architecture) - 2:
                output = tf.add(tf.matmul(self.layer_outputs[layer - 1], weights), biases)
            else:
                output = tf.nn.relu(tf.add(tf.matmul(self.layer_outputs[layer - 1], weights), biases))

            self.layer_weights.append(weights)
            self.layer_biases.append(biases)
            self.layer_outputs.append(output)

        """self.w2 = tf.Variable(tf.truncated_normal(shape=(20, 20), mean=weight_mean))
        self.b2 = tf.Variable(tf.truncated_normal(shape=(1, 20), mean=weight_mean))
        self.l2 = tf.nn.relu(tf.add(tf.matmul(self.l1, self.w2), self.b2))

        self.w3 = tf.Variable(tf.truncated_normal(shape=(20, output_length), mean=weight_mean))
        self.b3 = tf.Variable(tf.truncated_normal(shape=(1, output_length), mean=weight_mean))"""

        self.tf_out = tf.reshape(self.layer_outputs[-1], shape=(architecture[-1], ))

        self.tf_loss = tf.losses.mean_squared_error(self.tf_y, self.tf_out)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.tf_loss)

        self.tf_out_summary = tf.summary.histogram("Q values", self.tf_out)

        self.copy_operation = None

    def predict(self, tf_session, input, summary=False):
        if summary:
            output, summary = tf_session.run([self.tf_out, self.tf_out_summary], feed_dict={self.tf_x: input})
            return output, summary
        else:
            return tf_session.run([self.tf_out], feed_dict={self.tf_x: input})[0]

    def train(self, tf_session, input, output):
        loss, _ = tf_session.run([self.tf_loss, self.optimizer], feed_dict={self.tf_x: input, self.tf_y: output})
        return loss

    def assign_weights_from(self, tf_session, source_dnn):

        if self.copy_operation is None:
            ops = []
            for layer in range(len(self.layer_weights)):
                ops.append(self.layer_weights[layer].assign(source_dnn.layer_weights[layer]))
                ops.append(self.layer_biases[layer].assign(source_dnn.layer_biases[layer]))
            """ops.append(self.w1.assign(source_dnn.w1))
            ops.append(self.w2.assign(source_dnn.w2))
            ops.append(self.w3.assign(source_dnn.w3))
            ops.append(self.b1.assign(source_dnn.b1))
            ops.append(self.b2.assign(source_dnn.b2))
            ops.append(self.b3.assign(source_dnn.b3))"""

            self.copy_operation = tf.group(ops)

        tf_session.run(self.copy_operation)


