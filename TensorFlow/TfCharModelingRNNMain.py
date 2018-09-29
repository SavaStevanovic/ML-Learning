import numpy as np
import TensorFlow.TfCharRnn as rnn


with open('./Data/Hamlet.txt', 'r', encoding='utf-8') as f:
    text = f.read()

text = text[15858:]
chars = set(text)
char2int = {ch: i for i, ch in enumerate(chars)}
int2char = dict(enumerate(chars))
text_ints = np.array([char2int[ch] for ch in text], dtype=np.int32)


def reshape_data(sequence, batch_size, num_steps):
    tot_batch_length = batch_size*num_steps
    num_batches = int(len(sequence)/tot_batch_length)
    if num_batches*tot_batch_length+1 > len(sequence):
        num_batches = num_batches-1

    x = sequence[0:num_batches*tot_batch_length]
    y = sequence[1:num_batches*tot_batch_length+1]

    x_batch_splits = np.split(x, batch_size)
    y_batch_splits = np.split(y, batch_size)

    x = np.stack(x_batch_splits)
    y = np.stack(y_batch_splits)

    return x, y


batch_size = 64
num_step = 100
train_x, train_y = reshape_data(text_ints, batch_size, num_step)

rnn1=rnn.CharRNN(num_classes=len(chars),batch_size=batch_size)
rnn1.train(train_x,train_y,num_epochs=200,ckpt_dir='./model-200/')

del rnn1

np.random.seed(123)
rnn1 = rnn.CharRNN(num_classes=len(chars), sampling=True)
print(rnn1.sample(chars, char2int, int2char,
                  output_length=500, ckpt_dir='./model-200/'))
