import pandas as pd
import numpy as np
import pyprind
from collections import Counter
from string import punctuation
import TensorFlow.TfSentimentRNN as rnn

df = pd.read_csv('./ReviewData/movie_data.csv', encoding='utf-8')
counts = Counter()
pbar = pyprind.ProgBar(len(df['review']), title='Counting words occurrances')

for i, review in enumerate(df['review']):
    text = ''.join([c if c not in punctuation else ' ' +
                    c+' ' for c in review]).lower()
    df.loc[i, 'review'] = text
    pbar.update()
    counts.update(text.split())

word_counts = sorted(counts, key=counts.get, reverse=True)
print(word_counts[:10])
word_to_int = {word: ii for ii, word in enumerate(word_counts, 1)}

mapped_reviews = []
pbar = pyprind.ProgBar(len(df['review']), title='Map reviews to ints')

for review in df['review']:
    mapped_reviews.append([word_to_int[word] for word in review.split()])
    pbar.update()

sequence_length = 200
sequences = np.zeros((len(mapped_reviews), sequence_length), dtype=int)

for i, row in enumerate(mapped_reviews):
    review_arr = np.array(row)
    sequences[i, -len(row):] = review_arr[-sequence_length:]

X_train = sequences[:25000, :]
y_train = df.loc[:25000, 'sentiment'].values
X_test = sequences[25000:, :]
y_test = df.loc[25000:, 'sentiment'].values

np.random.seed(123)

n_words = len(list(word_to_int.values()))+1

rnn1 = rnn.SentimentRNN(n_words=n_words, seq_len=sequence_length, embed_size=256,
                       lstm_size=128, num_layers=1, batch_size=100, learning_rate=0.001)

rnn1.train(X_train, y_train, num_epochs=40)

preds = rnn1.predict(X_test)
y_true = y_test[:len(preds)]
print('Test Acc.: %.3f' % (
    np.sum(preds == y_true) / len(y_true)))
