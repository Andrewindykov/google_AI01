import tensorflow as tf
import numpy as np
from tensorflow import keras

def print_hi(name):
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def main():
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
    ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
    model.fit(xs, ys, epochs=55)
    print(model.predict([10.0]))


if __name__ == '__main__':
    print_hi('PyCharm')
    main()


