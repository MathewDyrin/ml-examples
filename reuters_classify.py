import numpy as np
from keras.datasets import reuters
from keras import layers
from keras import models

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)


def decode_newswire(index):
    word_index = reuters.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()]
    )
    decoded_newswire = ' '.join(
        [reverse_word_index.get(i-3, '?') for i in train_data[index]]
    )
    return decoded_newswire


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# OR USE KERAS to_categorical()
one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)


network = models.Sequential()
network.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
network.add(layers.Dense(64, activation="relu"))
network.add(layers.Dense(46, activation="softmax"))

network.compile(
    optimizer="rmsprop",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

network.fit(
    partial_x_train, partial_y_train, epochs=9, batch_size=512, validation_data=(x_val, y_val)
)

results = network.evaluate(x_test, one_hot_test_labels)
print(results)
