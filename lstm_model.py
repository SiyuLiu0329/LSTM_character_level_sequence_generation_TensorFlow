import tensorflow as tf
from lstm_utils import *

class LSTM:
    def __init__(self):
        self.cells = []

    def add_layer(self, n_units):
        pass


class Cell:
    def __init__(self, n_hidden, input_tensor, n_output):
        n_input, batch_size = input_tensor.get_shape().as_list()
        print("Cell input tensor shape: (", n_input, ",", batch_size, ")")

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

    def step(self):
        step_input = tf.concat(values=[self.previous_activation, self.input_tensor], axis=0)
        forget_gate = tf.sigmoid(tf.matmul(self.w_forget_gate, step_input) + self.b_forget_gate)
        update_gate = tf.sigmoid(tf.matmul(self.w_update_gate, step_input) + self.b_update_gate)
        output_gate = tf.tanh(tf.matmul(self.w_output_gate, step_input) + self.b_output_gate)
        assert forget_gate.get_shape() == self.previous_memory.get_shape() and update_gate.get_shape() == self.previous_memory.get_shape()


if __name__ == "__main__":
    lstm = LSTM()
    tensor = tf.get_variable('input', shape=[27, 100])
    cell = Cell(512, tensor, 27)
    cell.step()