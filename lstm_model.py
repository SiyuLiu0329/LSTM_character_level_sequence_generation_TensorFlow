import tensorflow as tf

class LSTM:
    def __init__(self):
        self.cells = []

    def add_layer(self, n_units):
        pass


class Cell:
    def __init__(self, n_hidden, input_tensor, n_output):
        n_input, batch_size, n_time_steps = input_tensor.get_shape().as_list()

        initialiser = tf.contrib.layers.xavier_initializer()

        self.w_forget_gate = tf.get_variable('w_forget_gate', shape=[n_hidden, n_hidden + n_input], initializer=initialiser)
        self.w_update_gate = tf.get_variable('w_update_gate', shape=[n_hidden, n_hidden + n_input], initializer=initialiser)
        self.w_output_gate = tf.get_variable('w_output_gate', shape=[n_hidden, n_hidden + n_input], initializer=initialiser)
        self.w_input = tf.get_variable('w_input', shape=[n_hidden, n_hidden + n_input], initializer=initialiser)
        self.w_output = tf.get_variable('w_output', shape=[n_output, n_hidden], initializer=initialiser)

        self.b_forget_gate = tf.get_variable('b_forget_gate', shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_update_gate = tf.get_variable('b_update_gate', shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_output_gate = tf.get_variable('b_output_gate', shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_input = tf.get_variable('b_input', shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_output = tf.get_variable('b_output', shape=[n_output, 1], initializer=tf.zeros_initializer())

        self.previous_activation = tf.get_variable('previous_activation', shape=[n_hidden, batch_size], initializer=tf.zeros_initializer())
        self.previous_memory = tf.get_variable('previous_memory', shape=[n_hidden, batch_size], initializer=tf.zeros_initializer())

        self.input_tensor = input_tensor
        self.n_input = n_input
        self.batch_size = batch_size
        self.n_time_steps = n_time_steps
        self.t = 0

    def step(self):
        x_t = tf.slice(self.input_tensor, [0, 0, self.t], [self.n_input, self.batch_size, 1])
        x_t = tf.reshape(x_t, [self.n_input, self.batch_size])
        step_input = tf.concat(values=[self.previous_activation, x_t], axis=0)
        forget_gate = tf.sigmoid(tf.matmul(self.w_forget_gate, step_input) + self.b_forget_gate)
        update_gate = tf.sigmoid(tf.matmul(self.w_update_gate, step_input) + self.b_update_gate)
        update = tf.tanh(tf.matmul(self.w_output_gate, step_input) + self.b_output_gate)
        output_gate = tf.sigmoid(tf.matmul(self.w_output_gate, step_input) + self.b_output_gate)
        self.previous_memory = tf.multiply(forget_gate, self.previous_memory) + tf.multiply(update_gate, update)
        self.previous_activation = tf.multiply(output_gate, tf.tanh(self.previous_memory))

        pred = tf.nn.softmax(tf.matmul(self.w_output, self.previous_activation) + self.b_output)
        self.t += 1
        return pred
        


if __name__ == "__main__":
    lstm = LSTM()
    tensor = tf.get_variable('input', shape=[27, 100, 60])
    cell = Cell(512, tensor, 27)
    pred = cell.step()
    print(pred, pred.get_shape())