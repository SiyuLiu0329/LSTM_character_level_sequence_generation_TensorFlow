import tensorflow as tf
import random
import string
import numpy as np

RNN_SIZE = 64

class Cell:
    """A single LSTM cell with pre-initialised weights and biases.

     Attributes:
        self._w_forget_gete: weights applied to the forget gate.
        self._w_update_gate: weights applied the update gate.
        self._w_tanh: weights applied the first tanh function.
        self._w_output_gate: weights applied to the output gate.
        self._w_out: weights applied to the softmax output function.

        self._b_forget_gete: biases applied to the forget gate.
        self._b_update_gate: biases applied the update gate.
        self._b_tanh: biases applied the first tanh function.
        self._b_output_gate: biases applied to the output gate.
        self._b_out: biases applied to the softmax output function.

        self._state: activation from the previous step.
        self._c:  memory from the previous step.
        self._input_tensor: a reference to the input tensor.

    """
    def __init__(self, n_units, n_output, input_tensor):
        """Inits cell

        Args:
            n_units (int): size of the hidden states.
            n_output (int): size of the output of the cell.
            input_tensor (obj: `Tensor`): input to the cell.
        """
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
        """Forward pass.
        Perform a forward pass and update the stored states.
        Returns:
            pred (obj `Tensor`): the output tensor of this forward pass.
            logits (obj `Tensor`): logits used by the softmax function (later used to compute loss).
        """
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
        """Get all weights and biases in the cell.
        Returns:
            elf._w_forget_gete: weights applied to the forget gate.
            self._w_update_gate: weights applied the update gate.
            self._w_tanh: weights applied the first tanh function.
            self._w_output_gate: weights applied to the output gate.
            self._w_out: weights applied to the softmax output function.

            self._b_forget_gete: biases applied to the forget gate.
            self._b_update_gate: biases applied the update gate.
            self._b_tanh: biases applied the first tanh function.
            self._b_output_gate: biases applied to the output gate.
            self._b_out: biases applied to the softmax output function.
        """
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
    """Opens a file and return the content of the file.
    Args:
        path (str): path of the file.
    Returns:
        data (str): content of the file.
    """
    fd = open(path, 'r', encoding='utf-8')
    data = fd.read()
    return data

def process_text(data):
    """Pre-processes read data
    Split up data into words and convert them into one hot vectors of the same size, 
    which will be used to construct the x(inputs) and y(labels).

    Args:
        data (str): data read from a file.

    Returns:
        x_train: training set x.
        y_train: training set y.
        char_to_ix (obj: `dict`): dictionary mapping from characters to indices.
        ix_to_char (obj: `dict`): dictionary mapping from indices to characters.
    """
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
    assert np.argmax(x_train[3 + 1][0]) == np.argmax(y_train[3][0])
    return x_train, y_train, char_to_ix, ix_to_char


def prepare_training_set(data, num_chars=27):
    """(Not finished) Divide the data set into small batches
    Args: 
        data: a numpy array of data
        num_chars (int): number of unique characters appears in the text file
    Returns:
        a numpy array of batches of data
        (Not finished, currently only supports stochastic gradient descent)
    """
    batches = []
    for i in range(len(data[0])):
        batches.append(data[:, i, :].reshape((-1, 1, num_chars)))
    return batches

def sigmoid(x):    
  return 1 / (1 + np.exp(-x))

def generate_sequence(weights, ix_to_char, max_len = 30, num_chars=27):
    """Generates a new word
    Takes the nerual network's weights and biases as inputs to generate new sequences. If a new line character appears
    the model will stop generating and prints the sequence generated.
    Args:
        weights: list of weights and biases extracted from the lstm model
        ix_to_char (obj: `dict`): a dictionary mapping from indices to characters
        max_len (int): maximum sequence length
        num_chars (int): number of unique characters in the text file
    """
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
    """Simple softmax function
    """
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def build_model(cell, labels, n_time_steps=28, lr= 0.002):
    """Build a model using the lstm cell.
    A helper function to build the conputation graph over time (unrolling) steps using the lstm cell.
    Args:
        cell (obj: `Cell`): a single lstm cell object
        labels: labels in the training set
        lr: learning rate
    """
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
    data = read_file('txt/drugs.txt')
    x, y, char_to_ix, ix_to_char = process_text(data)
    len_max, size, num_chars = x.shape
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
            idx = i % size
            sess.run(optimise_opt, feed_dict={input_tensor: x_train[idx], labels: y_train[idx]})
            if i % 500 == 0:
                print('\niter=' + str(i))
                print(sess.run(total_loss, feed_dict={input_tensor: x_train[idx], labels: y_train[idx]}))
                weights = cell.get_weights()
                weights = sess.run(weights)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
                generate_sequence(weights, ix_to_char, max_len = len_max, num_chars=num_chars)
            i += 1
