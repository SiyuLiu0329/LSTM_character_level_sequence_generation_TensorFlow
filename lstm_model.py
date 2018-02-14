import tensorflow as tf
import random
import string


class Cell:
    def __init__(self, n_hidden, input_tensor, n_output):
        self.n_input, self.batch_size, self.n_time_steps = input_tensor.get_shape().as_list()

        initialiser = tf.contrib.layers.xavier_initializer()
        cell_id = self._generate_cell_id()
        self.n_output = n_output

        self.w_forget_gate = tf.get_variable('w_forget_gate_' + cell_id, shape=[n_hidden, n_hidden + self.n_input], initializer=initialiser)
        self.w_update_gate = tf.get_variable('w_update_gate_' + cell_id, shape=[n_hidden, n_hidden + self.n_input], initializer=initialiser)
        self.w_output_gate = tf.get_variable('w_output_gate_' + cell_id, shape=[n_hidden, n_hidden + self.n_input], initializer=initialiser)
        self.w_input = tf.get_variable('w_input_' + cell_id, shape=[n_hidden, n_hidden + self.n_input], initializer=initialiser)

        self.b_forget_gate = tf.get_variable('b_forget_gate_' + cell_id, shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_update_gate = tf.get_variable('b_update_gate_' + cell_id, shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_output_gate = tf.get_variable('b_output_gate_' + cell_id, shape=[n_hidden, 1], initializer=tf.zeros_initializer())
        self.b_input = tf.get_variable('b_input_' + cell_id, shape=[n_hidden, 1], initializer=tf.zeros_initializer())

        self.b_output = tf.get_variable('b_output_' + cell_id, shape=[n_output, 1], initializer=tf.zeros_initializer())
        self.w_output = tf.get_variable('w_output_' + cell_id, shape=[n_output, n_hidden], initializer=initialiser)

        self.previous_activation = tf.get_variable('previous_activation_' + cell_id, shape=[n_hidden, self.batch_size], initializer=tf.zeros_initializer())
        self.previous_memory = tf.get_variable('previous_memory_' + cell_id, shape=[n_hidden, self.batch_size], initializer=tf.zeros_initializer())

        self.input_tensor = input_tensor
        self.t = 0

    def _generate_cell_id(self): 
        return ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(5))

    def step(self):
        assert self.t <= self.n_time_steps - 1
        x_t = tf.slice(self.input_tensor, [0, 0, self.t], [self.n_input, self.batch_size, 1])
        x_t = tf.reshape(x_t, [self.n_input, self.batch_size])

        step_input = tf.concat(values=[self.previous_activation, x_t], axis=0)
        forget_gate = tf.sigmoid(tf.matmul(self.w_forget_gate, step_input) + self.b_forget_gate)
        update_gate = tf.sigmoid(tf.matmul(self.w_update_gate, step_input) + self.b_update_gate)
        update = tf.tanh(tf.matmul(self.w_output_gate, step_input) + self.b_output_gate)
        output_gate = tf.sigmoid(tf.matmul(self.w_output_gate, step_input) + self.b_output_gate)
        self.previous_memory = tf.multiply(forget_gate, self.previous_memory) + tf.multiply(update_gate, update)
        self.previous_activation = tf.multiply(output_gate, tf.tanh(self.previous_memory))
        self.t += 1

        output = tf.nn.softmax(tf.matmul(self.w_output, self.previous_activation) + self.b_output)
        return output



if __name__ == "__main__":
    lstm = LSTM()
    tensor = tf.get_variable('input', shape=[27, 100, 60])
    cell = Cell(512, tensor, 27)
    print(cell.step())
    print(cell.step())
    print(cell.step())