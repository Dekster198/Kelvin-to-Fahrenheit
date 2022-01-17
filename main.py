import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.layers import Dense

k = np.array([-100, -70, -10, 15, 42, 45, 50, 80, 90, 95])
f = np.array([-639.67, -585.67, -477.67, -432.67, -384.07, -378.67, -369.67, -315.67, -297.67, -288.67])

model = keras.Sequential()
model.add(Dense(units=1, input_shape=(1,), activation='linear'))
model.compile(loss='mean_squared_error', optimizer=keras.optimizers.Adam(0.1))
history = model.fit(k, f, epochs=8500, verbose=False)

print(model.predict([-273]))
print(model.get_weights())

plt.plot(history.history['loss'])
plt.grid(True)
plt.show()