
# Example 4.20 - SimpleRNN2.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Input, Dense, SimpleRNN

# Mengubah data menjadi dataset
def convertToMatrix(data, step):
    X, Y = [], []
    for i in range(len(data) - step):
        d = i + step
        X.append(data[i:d])
        Y.append(data[d])
    return np.array(X), np.array(Y)

# Membaca dataset jumlah penumpang pesawat
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv"
df = pd.read_csv(url, usecols=[1], engine="python")

plt.plot(df)
plt.title("Jumlah Penumpang Pesawat")
plt.show()

# Membagi data menjadi data latih dan data uji
step = 4
N = df.shape[0]
Tp = int(N * 0.8)

values = df.values
train, test = values[:Tp, :], values[Tp:, :]

# Menambahkan data untuk prediksi
test = np.append(test, np.repeat(test[-1], step))
train = np.append(train, np.repeat(train[-1], step))

trainX, trainY = convertToMatrix(train, step)
testX, testY = convertToMatrix(test, step)

# Menyesuaikan bentuk input untuk SimpleRNN
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# Membuat model SimpleRNN
model = Sequential()
model.add(Input(shape=(1, step)))
model.add(SimpleRNN(units=32, activation="relu"))
model.add(Dense(8, activation="relu"))
model.add(Dense(1))

model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.summary()

# Melatih model
model.fit(trainX, trainY, epochs=10, batch_size=16, verbose=2)

# Melakukan prediksi
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

predicted = np.concatenate((trainPredict, testPredict), axis=0)

# Menghitung error data latih
trainScore = model.evaluate(trainX, trainY, verbose=0)
print("Training Loss:", trainScore)

# Menampilkan hasil prediksi
plt.figure(figsize=(10, 5))
plt.plot(df.values, label="Data Aktual")
plt.plot(np.arange(len(predicted)), predicted, label="Hasil Prediksi")
plt.axvline(Tp, color="red", label="Batas Data Latih dan Uji")
plt.title("Prediksi Jumlah Penumpang Pesawat dengan SimpleRNN")
plt.xlabel("Indeks Waktu")
plt.ylabel("Jumlah Penumpang")
plt.legend()
plt.show()
