

import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Bidirectional, Dropout, Dense,GRU

data_train = pd.read_csv(r"D:\cybersecurity\unbarchive\UNSW_NB15_train.csv")
data_train

data_test = pd.read_csv(r"D:\cybersecurity\unbarchive\UNSW_NB15_test.csv")

data_train = data_train.drop('attack_cat', axis=1)
data_test = data_test.drop('attack_cat', axis=1)

data_train["state"].unique()
data_test["state"].unique()

data_test.drop(data_test[data_test['state'] == "CLO"].index, inplace=True)
data_test.drop(data_test[data_test['state'] == "ACC"].index, inplace=True)
data_test["state"].unique()

df = data_train.iloc[:, 2: ]
df_test = data_test.iloc[:, 2: ]


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df.iloc[: , 3:-1] = scaler.fit_transform(df.iloc[:,3:-1])
df_test.iloc[:, 3:-1] = scaler.transform(df_test.iloc[:, 3:-1])

df_scaled = df.to_numpy()

df_scaled_test = df_test.to_numpy()




print(data_train[data_train["proto"]=="udp"])
print(df_scaled_test[:,0])

from sklearn.preprocessing import LabelEncoder
label_encoder1 = LabelEncoder()
label_encoder2 = LabelEncoder()
label_encoder3 = LabelEncoder()

df_scaled[:,0] = label_encoder1.fit_transform(df_scaled[:,0])
df_scaled_test[:, 0] = label_encoder1.transform(df_scaled_test[:, 0])

df_scaled[:,1] = label_encoder2.fit_transform(df_scaled[:,1])
df_scaled_test[:, 1] = label_encoder2.transform(df_scaled_test[:, 1])

df_scaled[:,2] = label_encoder3.fit_transform(df_scaled[:,2])
df_scaled_test[:, 2] = label_encoder3.transform(df_scaled_test[:, 2])

#df_scaled[:,-1] = label_encoder.fit_transform(df_scaled[:,-1])





df_scaled = df_scaled.astype(float)
df_test = df_scaled_test.astype(float)



x_train = []
y_train = []


for i in range(100,len(df_scaled)):
    x_train.append(df_scaled[i-100:i,0:df_scaled.shape[1]])
    y_train.append(df_scaled[i:i+1,41])

x_train , y_train = np.array(x_train), np.array(y_train)



x_test = []
y_test = []


for i in range(100, len(df_test)):
    x_test.append(df_test[i-100:i, 0:df_test.shape[1]])
    y_test.append(df_test[i:i+1,41])  

x_test = np.array(x_test)
y_test = np.array(y_test)














model = Sequential()
model.add(LSTM(units = 40, return_sequences=True, input_shape = (x_train.shape[1], x_train.shape[2])))

model.add(Bidirectional(LSTM(40, return_sequences=False)))

model.add(Dropout(0.1))

model.add(Dense(40, activation='relu'))  
model.add(Dense(1, activation='sigmoid'))  

model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()




model.fit(x_train, y_train, epochs = 10, batch_size = 32)















predictions = model.predict(x_test)
from sklearn.metrics import accuracy_score

predictions_binary = (predictions > 0.5).astype(int)

accuracy = accuracy_score(y_test, predictions_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")
