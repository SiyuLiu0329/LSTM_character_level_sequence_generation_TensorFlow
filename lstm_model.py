import tensorflow as tf
import random
import string
import numpy as np

RNN_SIZE = 256

class Cell:
    def __init__(self, n_units, n_output, input_tensor):
        _, batch_size, n_input = input_tensor.get_shape()
        self._w_forget_gete = tf.Variable(np.random.randn(n_input + n_units, n_units), dtype=tf.float64)
        self._w_update_gate = tf.Variable(np.random.randn(n_input + n_units, n_units), dtype=tf.float64)
        self._w_tanh = tf.Variable(np.random.randn(n_input + n_units, n_units), dtype=tf.float64)
        self._w_output_gate = tf.Variable(np.random.rand(n_input + n_units, n_units), dtype=tf.float64)
        self._w_out = tf.Variable(np.random.randn(n_units, n_output), dtype=tf.float64)

        self._b_forget_gate = tf.Variable(np.random.randn(1, n_units), dtype=tf.float64)
        self._b_update_gate = tf.Variable(np.random.randn(1, n_units), dtype=tf.float64)
        self._b_tanh = tf.Variable(np.random.randn(1, n_units), dtype=tf.float64) 
        self._b_output_gate = tf.Variable(np.random.randn(1, n_units), dtype=tf.float64)
        self._b_out = tf.Variable(np.random.randn(1, n_output), dtype=tf.float64)

        self._state = tf.constant(np.zeros((batch_size, n_units)), dtype=tf.float64)
        self._c = tf.constant(np.zeros((batch_size, n_units)), dtype=tf.float64)
        self._input_tensor = input_tensor
        

        self._t = 0

    def forward_pass(self):
        # change to lstm later
        current_input = self._input_tensor[self._t]
        current_input = tf.concat([self._state, current_input], axis=1)
        forget_gate = tf.sigmoid(tf.matmul(current_input, self._w_forget_gete) + self._b_forget_gate)
        update_gate = tf.sigmoid(tf.matmul(current_input, self._w_update_gate) + self._b_update_gate)
        tanh = tf.tanh(tf.matmul(current_input, self._w_tanh) + self._b_tanh)
        self._c = tf.multiply(forget_gate, self._c) + tf.multiply(update_gate, tanh)
        output_gate = tf.sigmoid(tf.matmul(current_input, self._w_output_gate) + self._b_output_gate)
        self._state = tf.multiply(output_gate, tf.tanh(self._c))

        logits = tf.matmul(self._state, self._w_out) + self._b_out
        pred = tf.nn.softmax(logits)
        self._t += 1
        return pred, logits

    def get_weights(self):
        return (
                self._w_forget_gete, 
                self._w_update_gate,
                self._w_tanh,
                self._w_output_gate,
                self._w_out,
                self._b_forget_gate,
                self._b_update_gate,
                self._b_tanh,
                self._b_output_gate,
                self._b_out
               )


def read_file(path):
    fd = open(path, 'r', encoding='utf-8')
    data = fd.read()
    return data

def process_text(data):
    data = data.lower()
    chars = list(set(data))
    lines = data.split()
    np.random.shuffle(lines)
    char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
    ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
    max_len = 0
    for line in lines:
        if len(line) > max_len:
            max_len = len(line)
    n_time_steps = max_len + 2 # 1 for the None character at the start the other for the '\n' character at the end

    x_train = np.zeros((n_time_steps, len(lines), len(chars))) 
    y_train = np.zeros((n_time_steps, len(lines), len(chars))) 

    for i in range(len(lines)):
        for j in range(len(lines[i])):
            idx = char_to_ix[lines[i][j]]
            y_train[j][i][idx] = 1
            x_train[j + 1][i][idx] = 1
        y_train[j + 1][i][0] = 1
        x_train[j + 2][i][0] = 1
    assert ix_to_char[np.argmax(x_train[3 + 1][0])] == ix_to_char[np.argmax(y_train[3][0])]
    return x_train, y_train, char_to_ix, ix_to_char


def prepare_training_set(data, num_chars=27):
    batches = []
    for i in range(len(data[0])):
        batches.append(data[:, i, :].reshape((-1, 1, num_chars)))
    return batches

def sigmoid(x):
  return 1 / (1 + np.exp(-x))

def generate_sequence(weights, ix_to_char, max_len = 30, num_chars=27):
    (
        w_forget, w_update, w_tanh, w_output, w_out,
        b_forget, b_update, b_tanh, b_output, b_out,       
    ) = weights

    state = np.zeros((1, RNN_SIZE))
    c = np.zeros((1, RNN_SIZE))
    current_input = np.zeros((1, num_chars))
    seq = []
    idx = -1
    while len(seq) <= max_len:

        current_input = np.concatenate((state, current_input), axis=1)
        forget_gate = sigmoid(np.dot(current_input, w_forget) + b_forget)
        update_gate = sigmoid(np.dot(current_input, w_update) + b_update)
        tanh = np.tanh(np.dot(current_input, w_tanh) + b_tanh)
        c = np.multiply(forget_gate, c) + np.multiply(update_gate, tanh)
        output_gate = sigmoid(np.dot(current_input, w_output) + b_output)
        state = np.multiply(output_gate, np.tanh(c))
        logits = np.dot(state, w_out) + b_out
        pred = softmax(logits)

        idx = np.random.choice([i for i in range(len(ix_to_char))], p = pred.ravel())
        if idx == 0:
            break
        current_input = np.zeros((1, len(ix_to_char)))
        current_input[0][idx] = 1
        seq.append(idx)
    
    seq = [ix_to_char[ix] for ix in seq]
    print(''.join(seq))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def build_model(cell, labels, n_time_steps=28, lr= 0.002):
    losses = []
    for i in range(n_time_steps):
        _, logits = cell.forward_pass()
        label = labels[i, :, :]
        losses.append(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = lr
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       2000, 0.96, staircase=True)
    total_loss = tf.reduce_sum(losses)
    optimiser = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimiser.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 2.0)
    optimise = optimiser.apply_gradients(zip(gradients, variables), global_step=global_step)
    return optimise, total_loss

if __name__ ==  '__main__':
    data = read_file('txt/names.txt')
    x, y, char_to_ix, ix_to_char = process_text(data)
    len_max, _, num_chars = x.shape
    x_train = prepare_training_set(x, num_chars=num_chars)
    y_train = prepare_training_set(y, num_chars=num_chars)
    input_tensor = tf.placeholder(tf.float64, shape=[None, 1, num_chars])
    cell = Cell(RNN_SIZE, num_chars, input_tensor)
    labels = tf.placeholder(tf.float64, [None, 1, len(ix_to_char)])
    optimise_opt, total_loss = build_model(cell, labels, n_time_steps=len_max)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        while True:

            
            idx = i % 1536
            sess.run(optimise_opt, feed_dict={input_tensor: x_train[idx], labels: y_train[idx]})
            if i % 500 == 0:
                print('\niter=' + str(i * 500))
                print(sess.run(total_loss, feed_dict={input_tensor: x_train[idx], labels: y_train[idx]}))
                weights = cell.get_weights()
                weights = sess.run(weights)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
            i += 1