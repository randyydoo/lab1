import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split

def main():
    x_vals = np.linspace(-100, 100, 10000)
    y_vals = x_vals * np.sin((x_vals**2)/300)

      
    data = np.column_stack((x_vals, y_vals))
    np.savetxt('dataset.csv', data, delimiter=',', header='x,y', comments='')

    data = np.array(data)
    x_train, x_test, y_train, y_test = train_test_split(x_vals, y_vals, test_size=0.60)
    
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    model1.add(Input(shape=(1)))
    model1.add(Dense(16, activation=None))
    model1.add(Dense(8, activation=None))
    model1.add(Dense(4, activation=None))

    model1.compile(optimizer="adam", loss="MeanSquaredError")
    model1.fit(x=x_train, y=y_train, validation_split = 0.4, epochs=2)
    result1 = model1.evaluate(x=x_test, y=y_test)

    model2.compile(optimizer="adam", loss="mean_absolute_error")
    model2.fit(x=x_train, y=y_train, validation_split = 0.4, epochs=2)
    result2 = model2.evaluate(x=x_test, y=y_test)

    model3.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    model3.fit(x=x_train, y=y_train, validation_split = 0.4, epochs=2)
    result3 = model3.evaluate(x=x_test, y=y_test)

    print([result1, result2, result3])

if __name__ == "__main__":
    main()
