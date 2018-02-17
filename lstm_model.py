import tensorflow as tf
import random
import string
import numpy as np



class Cell:
    def __init__(self, n_units, n_output, input_tensor):
        _, batch_size, n_input = input_tensor.get_shape()
        self._w = tf.Variable(np.random.randn(n_input + n_units, n_units), dtype=tf.float64)
        self._b = tf.Variable(np.random.randn(1, n_units)) 

        self._state = tf.constant(np.zeros((batch_size, n_units)), dtype=tf.float64)
        self._input_tensor = input_tensor

        self._w_out = tf.Variable(np.random.randn(n_units, n_output), dtype=tf.float64)
        self._b_out = tf.Variable(np.random.randn(1, n_output), dtype=tf.float64)

        self._t = 0

    def forward_pass(self):
        # change to lstm later
        current_input = self._input_tensor[self._t]
        current_input = tf.concat([self._state, current_input], axis=1)
        self._state = tf.tanh(tf.matmul(current_input, self._w) + self._b)    
        logits = tf.matmul(self._state, self._w_out) + self._b_out
        pred = tf.nn.softmax(logits)
        self._t += 1
        return pred, logits

    def get_weights(self):
        return self._w, self._b, self._w_out, self._b_out



def read_file(path):
    fd = open(path, 'r')
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


def to_batches(data, batch_size):
    batches = []
    for i in range(1536):
        batches.append(data[:, i, :].reshape((-1, 1, 27)))
    return batches
    

def generate_sequence(w, b, w_out, b_out, ix_to_char):
    state = np.zeros((1, 256))
    current_input = np.zeros((1, 27))
    seq = []
    idx = -1
    while len(seq) <= 30:
        if idx == 0:
            break
        current_input = np.concatenate((state, current_input), axis=1)
        state = np.tanh(np.dot(current_input, w) + b)
        logits = np.dot(state, w_out) + b_out
        pred = softmax(logits)
        idx = np.random.choice([i for i in range(27)], p = pred.ravel())
        current_input = np.zeros((1, 27))
        current_input[0][idx] = 1
        seq.append(idx)
    
    seq = [ix_to_char[ix] for ix in seq]
    print(''.join(seq))

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


if __name__ ==  '__main__':
    data = read_file('txt/dinos.txt')
    x_train, y_train, char_to_ix, ix_to_char = process_text(data)
    x_batches, y_batches = to_batches(x_train, 1), to_batches(y_train, 1)

    input_tensor = tf.placeholder(tf.float64, shape=[28, 1, 27])
    # input_tensor = tf.constant([[[1,2,3,4,5], [6,7,8,9,10]], [[1.1,2,3,4,5], [6,7,8,9,10]]], shape=[2, 2, 5], dtype=tf.float64)
    cell = Cell(256, 27, input_tensor)

    losses = []
    labels = tf.placeholder(tf.float64, [None, 1, 27])
    for i in range(28):
        pred, logits = cell.forward_pass()
        label = labels[i, :, :]
        losses.append(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=label))

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.002
    learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                       2000, 0.96, staircase=True)
    total_loss = tf.reduce_sum(losses)
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients, variables = zip(*optimizer.compute_gradients(total_loss))
    gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
    optimize = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        i = 0
        while True:
            idx = i % 1536
            sess.run(optimize, feed_dict={input_tensor: x_batches[idx], labels: y_batches[idx]})
            if i % 100 == 0:
                print('iter=' + str(i))
                print(sess.run(total_loss, feed_dict={input_tensor: x_batches[idx], labels: y_batches[idx]}))
                w, b, w_out, b_out = cell.get_weights()
                w, b, w_out, b_out = sess.run([w, b, w_out, b_out])
                generate_sequence(w, b, w_out, b_out, ix_to_char)

            i += 1
