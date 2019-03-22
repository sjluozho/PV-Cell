from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.wrappers.scikit_learn import KerasRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


seed = 42


# dataset split
def split():
    feature_name = pd.read_csv('../Database/first_60_features.csv')
    feature_1k = pd.read_csv(
                 '../Database/No_Missing_Value/features_25000_26000_R.csv')
    feature_name = feature_name.drop(columns='Unnamed: 0')
    selected_features = feature_name.values.reshape(1, -1)
    features = feature_1k[selected_features[0]]
    fe_list = features.columns.tolist()

    X = features[fe_list].values
    Y = feature_1k[['pce']].values

    X_train_pn, X_test_pn, y_train, y_test = train_test_split(X, Y,
                                                              test_size=0.20,
                                                              random_state=seed
                                                              )
    return X_train_pn, X_test_pn, y_train, y_test

# normalize input dataset


def normalize(X_train_pn, X_test_pn):
    X_train_scaler = StandardScaler().fit(X_train_pn)
    X_train = X_train_scaler.transform(X_train_pn)
    X_test = X_train_scaler.transform(X_test_pn)
    return X_train, X_test


def nnmodel():
    # assemble the structure
    model = Sequential()
    model.add(Dropout(0.2))
    model.add(Dense(60, input_dim=60,
                    kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(15, kernel_initializer='normal', activation='relu'))
    model.add(Dense(8, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # compile the model
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


def model():
    # training
    np.random.seed(seed)
    X_train_pn, X_test_pn, y_train, y_test = split()
    X_train, X_test = normalize(X_train_pn, X_test_pn)
    estimator = KerasRegressor(build_fn=nnmodel,
                               epochs=2000, batch_size=400, verbose=0)
    history = estimator.fit(X_train, y_train, validation_split=0.25,
                            epochs=2000, batch_size=200, verbose=0)
    # display evaluation
    print("final MSE for train is %.2f and for validation is %.2f" %
          (history.history['loss'][-1], history.history['val_loss'][-1]))

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    test_loss = estimator.model.evaluate(X_test, y_test)
    print("test set mse is %.2f" % test_loss)

    plt.scatter(y_test, estimator.predict(X_test), s=4)
    plt.scatter(y_train, estimator.predict(X_train), s=3, c='r')
    plt.xlabel('Experimental PCE', size=15)
    plt.ylabel('Predict PCE', size=15)
    plt.legend(['Test', 'Train'])
    plt.plot(y_test, y_test, c='g')
    plt.show()

    SSR = ((y_test - estimator.predict(X_test)) ** 2).sum()
    SST = ((y_test - y_test.mean()) ** 2).sum()
    R_square = 1 - SSR/SST
    print("R-square of prediction is ", R_square)
    return estimator
