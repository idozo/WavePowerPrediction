import pandas as pd
import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop


def make_plot(predictions, test_labels, hours_ahead):
    plt.plot(list(range(0, len(test_labels))), test_labels, label='real results')
    plt.plot(list(range(0, len(predictions))), predictions, label='prediction')
    plt.legend()
    title = 'network with locations, ' + str(hours_ahead) + ' hours ahead'
    error = "MAE: {:.3f}, MSE: {:.3f}".format(mean_absolute_error(predictions, test_labels), mean_squared_error(predictions, test_labels))
    plt.title(title + '\n' + error)
    if save_graphs:
        plt.savefig('results/' + title +'.png')
    plt.show()
    mse_csv.at[hours_ahead, 'loc-net'] = mse
    mae_csv.at[hours_ahead, 'loc-net'] = mae

def plot_loss():
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.ylim(top=200, bottom=0)
    plt.legend(['Train', 'Val'], loc='upper right')
    plt.show()


def build_model(train_data):
    model = Sequential()
    model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1), input_shape=[train_data.shape[1]]))
    model.add(Dropout(0.3))
    model.add(Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    optimizer = RMSprop(0.001)
    # optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def norm(x, stat):
    return (x - stat['mean']) / stat['std']


def getData():
    train2017 = pd.read_csv("data/2017_input_loc.csv", skipinitialspace=True)
    train2018 = pd.read_csv("data/2018_input_loc.csv", skipinitialspace=True)
    train_dataset = pd.concat([train2017, train2018], ignore_index=True)
    # train_dataset = pd.read_csv("data/2018_input_loc.csv", skipinitialspace=True)
    test_dataset = pd.read_csv("data/2019_input_loc.csv", skipinitialspace=True)

    labels_cols = [str(hours_ahead) + 'AHEAD wavepower']
    input_cols = [col for col in train_dataset if 'AGO' in col]

    train_data = train_dataset[input_cols]
    train_labels = train_dataset[labels_cols]
    test_data = test_dataset[input_cols]
    test_labels = test_dataset[labels_cols]

    scaler = StandardScaler()
    scaler.fit(train_data)
    train_data = scaler.transform(train_data)
    test_data = scaler.transform(test_data)
    train_labels = train_labels.to_numpy()

    nrow = round(0.8 * train_data.shape[0])
    randomize = np.arange(len(train_data))
    np.random.shuffle(randomize)
    train_data = train_data[randomize]
    train_labels = train_labels[randomize]

    train_d = train_data[:nrow, :]
    val_d = train_data[nrow:, :]
    train_l = train_labels[:nrow]
    val_l = train_labels[nrow:]
    return train_d, train_l, val_d, val_l, test_data, test_labels

if __name__ == '__main__':
    hours_ahead_array = [1,2,5,10,20,24,30]
    save_graphs = True
    for hours_ahead in hours_ahead_array:
        mse_csv = pd.read_csv('results/mse_results.csv', index_col='hours ahead')
        mae_csv = pd.read_csv('results/mae_results.csv', index_col='hours ahead')
        train_data, train_labels, val_data, val_labels, test_data, test_labels = getData()
        model = build_model(train_data)
        EPOCHS = 300

        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(train_data, train_labels, epochs=300, validation_data=(val_data, val_labels), batch_size=64, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
        print()
        loss, mae, mse = model.evaluate(test_data, test_labels, verbose=2)
        print("Hours Ahead:{}, Testing set Mean Squred Error: {:5.2f}".format(hours_ahead, mse))

        test_predictions = model.predict(test_data).flatten()
        make_plot(test_predictions, test_labels, hours_ahead)
        mse_csv.to_csv('results/mse_results.csv', index='hours ahead')
        mae_csv.to_csv('results/mae_results.csv', index='hours ahead')