from keras.datasets import boston_housing
from keras import layers
from keras import models

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std


def build_network():
    network = models.Sequential()
    network.add(layers.Dense(64, activation="relu", input_shape=(train_data.shape[1],)))
    network.add(layers.Dense(64, activation="relu"))
    network.add(layers.Dense(1))
    network.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return network


k = 4
num_val_samples = len(train_data) // k
num_epochs = 500
all_scores = []
all_mae_histories = []

# for i in range(k):
#     print("processing fold #", i)
#     val_data = train_data[i * num_val_samples:(i+1) * num_val_samples]
#     val_targets = train_targets[i * num_val_samples:(i+1) * num_val_samples]
#
#     partial_train_data = np.concatenate(
#         [train_data[:i * num_val_samples], train_data[(i+1) * num_val_samples:]], axis=0
#     )
#
#     partial_train_targets = np.concatenate(
#         [train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0
#     )

network = build_network()

network.fit(
    train_data,
    train_targets,
    epochs=80,
    batch_size=16,
    verbose=0,
)

test_mse_score, test_mae_score = network.evaluate(test_data, test_targets)

print(test_mae_score)
