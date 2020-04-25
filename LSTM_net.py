import pandas as pd
import numpy as np
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import RMSprop

ש
def make_plot(predictions, test_labels, hours_ahead):
    plt.plot(list(range(0, len(test_labels))), test_labels, label='real results')
    plt.plot(list(range(0, len(predictions))), predictions, label='prediction')
    plt.legend()
    title = 'LSTM network, ' + str(hours_ahead) + ' hours ahead'
    error = "MAE: {:.3f}, MSE: {:.3f}".format(mean_absolute_error(predictions, test_labels), mean_squared_error(predictions, test_labels))
    plt.title(title + '\n' + error)
    if save_graphs:
        plt.savefig('results/' + title +'.png')
    plt.show()
    mse_csv.at[hours_ahead, 'LSTM-net'] = mse
    mae_csv.at[hours_ahead, 'LSTM-net'] = mae

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
    model.add(LSTM(input_shape=(hours_before, 18), units=(hours_before*18), return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.1)))
    model.add(Dropout(0.3))
    model.add(Dense(1))

    optimizer = RMSprop(0.001)
    model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
    return model


def getData():
    cols = ["VHM0", "VTM10", "VTM02", "VMDR", "VSDX", "VSDY", "VHM0_WW", "VTM01_WW", "VMDR_WW", "VHM0_SW1",
            "VTM01_SW1", "VMDR_SW1", "VHM0_SW2", "VTM01_SW2", "VMDR_SW2", "VPED", "VTPK", "wavepower"]


    train2017 = pd.read_csv("data/database_2017.csv", skipinitialspace=True)
    train2018 = pd.read_csv("data/database_2018.csv", skipinitialspace=True)
    df_train = pd.concat([train2017, train2018], ignore_index=True)
    # df_train = pd.read_csv("data/database_2018.csv")
    df_train = df_train[cols]
    df_test = pd.read_csv("data/database_2019.csv")
    df_test = df_test[cols]
    train_labels = df_train['wavepower'].to_numpy()[hours_before + hours_ahead:len(df_train) + hours_ahead]
    test_labels = df_test['wavepower'].to_numpy()[hours_before + hours_ahead:len(df_test) + hours_ahead]

    df_train = df_train.to_numpy()
    df_test = df_test.to_numpy()
    scaler = StandardScaler()
    scaler.fit(df_train)
    df_train = scaler.transform(df_train)
    df_test = scaler.transform(df_test)

    train = []
    for i in range(hours_before, df_train.shape[0] - hours_ahead):
        train.append(df_train[i-hours_before:i])
    train_data = np.array(train)

    test = []
    for i in range(hours_before, df_test.shape[0] - hours_ahead):
        test.append(df_test[i-hours_before: i])
    test_data = np.array(test)

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
    hours_before = 20
    # hours_before = 48
    hours_ahead_array = [1,2,5,10,20,24,30]
    # hours_ahead_array = [5]
    save_graphs = True
    for hours_ahead in hours_ahead_array:
        mse_csv = pd.read_csv('results/mse_results.csv', index_col='hours ahead')
        mae_csv = pd.read_csv('results/mae_results.csv', index_col='hours ahead')
        train_data, train_labels, val_data, val_labels, test_data, test_labels = getData()
        model = build_model(train_data)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(train_data, train_labels, epochs=300, validation_data=(val_data, val_labels), batch_size=64, verbose=0, callbacks=[early_stop, tfdocs.modeling.EpochDots()])
        loss, mae, mse = model.evaluate(test_data, test_labels, verbose=2)
        print("Hours Ahead:{}, Testing set Mean Squred Error: {:5.2f}".format(hours_ahead, mse))
        test_predictions = model.predict(test_data)
        make_plot(test_predictions, test_labels, hours_ahead)
        mse_csv.to_csv('results/mse_results.csv', index='hours ahead')
        mae_csv.to_csv('results/mae_results.csv', index='hours ahead')