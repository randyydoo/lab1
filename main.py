import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from sklearn.model_selection import train_test_split

def main():
    # create dataset
    x_vals = np.linspace(-100.0, 100.0, 1000000)
    x_vals = np.array(x_vals)
    y_vals = x_vals * np.sin((x_vals**2)/300)
      
    data = np.column_stack((x_vals, y_vals))
    np.savetxt('dataset.csv', data, delimiter=',', header='x,y', comments='')

    # Reshape x_vals into a 2D array
    x_vals_reshaped = x_vals.reshape(-1, 1)    
    
    #prepare data for training
    x_train, x_test, y_train, y_test = train_test_split(x_vals_reshaped, y_vals, test_size=0.60)
    
    model1 = Sequential()
    model2 = Sequential()
    model3 = Sequential()
    model1.add(Input(shape=(1)))
    model1.add(Dense(16, activation='relu'))
    model1.add(Dense(8, activation='relu'))
    model1.add(Dense(4, activation='relu'))
    
    model2.add(Input(shape=(1)))
    model2.add(Dense(16, activation='relu'))
    model2.add(Dense(8, activation='relu'))
    model2.add(Dense(4, activation='relu'))

    model3.add(Input(shape=(1)))
    model3.add(Dense(16, activation='relu'))
    model3.add(Dense(8, activation='relu'))
    model3.add(Dense(4, activation='relu'))
    
    test = [-20.9209209209209,-20.7207207207207,-20.5205205205205, -20.3203203203203, -20.1201201201201, -19.9199199199199]
    # config models
    model1.compile(optimizer="adam", loss="MeanSquaredError")
    model1.fit(x=x_train, y=y_train, epochs=2)
    print("Model 1 Predictions:", model1.predict(np.array(test)))
    result1 = model1.evaluate(x=x_test, y=y_test)

    model2.compile(optimizer="adam", loss="mean_absolute_error")
    model2.fit(x=x_train, y=y_train, epochs=2)
    print("Model 2 Predictions:", model2.predict(np.array(test)))
    result2 = model2.evaluate(x=x_test, y=y_test)

    model3.compile(optimizer="adam", loss="mean_absolute_percentage_error")
    model3.fit(x=x_train, y=y_train, epochs=2)
    print("Model 3 Predictions:", model3.predict(np.array(test)))
    result3 = model3.evaluate(x=x_test, y=y_test)
    
    # print([result1, result2, result3])

if __name__ == "__main__":
    main()

