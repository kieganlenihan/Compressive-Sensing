import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def scatter_plot(attr1, attr2):
    attribute1 = auto_data[attr1].to_numpy(dtype = float)
    attribute2 = auto_data[attr2].to_numpy(dtype = float)
    plt.scatter(attribute1, attribute2)
    font = {'fontname':'Times New Roman', 'size' : 14}
    plt.xlabel(attr1.capitalize(), **font)
    plt.ylabel(attr2.capitalize(), **font)
    plt.title(attr1.capitalize() + " vs " + attr2.capitalize(), **font)
    plt.show()
    # plt.savefig("%s_vs_%s_plot.png" % (attr1, attr2))
    plt.clf()
def model_test(X, Y, title, xlabel, ylabel, file_name):
    plt.scatter(X, Y, label = "Predicted Price")
    plt.plot(Y, Y, '-r', label = "Perfect Prediction")
    font = {'fontname':'Times New Roman', 'size' : 14}
    plt.title(title, **font)
    plt.xlabel(xlabel, **font)
    plt.ylabel(ylabel, **font)
    plt.legend()
    # plt.show()
    plt.savefig(file_name)
    plt.clf()
def tex_writer(attr1, attr2, count, hpics):
    print("\includegraphics[width=%.2f\linewidth]{%s_vs_%s_plot.png}" % (1/hpics, attr1, attr2), end = '')
    if count % hpics == 0:
        print("\\\[\\baselineskip]")
    else:
        print("\quad", end = '')
if __name__ == '__main__':
    file_path = '~/Documents/GitHub/ECE580/imports-85.data'
    raw_data = pd.read_csv(file_path, delimiter = ',', engine = 'python')
    ## Assign column names
    column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    raw_data.columns = column_names
    ## Preserve only relevant attributes
    important_cols = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    global auto_data
    auto_data = raw_data[important_cols]
    ## Remove unknown price instances
    with warnings.catch_warnings():
        warnings.simplefilter(action='ignore', category=FutureWarning)
        auto_data = auto_data[auto_data['price'] != '?']
    ## Replace '?' with NaN
    auto_data.replace('?', np.nan, inplace = True)
    ## Average out unknown instances and change data type to float
    for col in important_cols:
        auto_data[[col]] = auto_data[[col]].astype('float')
        mean = round(auto_data[col].mean(axis=0), 2)
        auto_data[col].replace(np.nan, mean, inplace=True)
    ## Plot each feature vs price
    # for attribute in important_cols:
    #     scatter_plot(attribute, 'price')
    ## Plot each feature against eachother
    count = 1
    for i in range(len(important_cols[:-1])):
        for j in range(i, len(important_cols[:-2])):
            # scatter_plot(important_cols[i], important_cols[j+1])
            # tex_writer(important_cols[i], important_cols[j+1], count, 3)
            count += 1
    ## Linear regression model based on engine-size
    lm = LinearRegression()
    X = auto_data[['engine-size']]
    Y = auto_data['price']
    lm.fit(X,Y)
    print("LinReg b = %f" %(lm.intercept_))
    print("LinReg a = %f" % (lm.coef_))
    print("LinReg r^2 = %f" % (lm.score(X, Y)))
    model_test(lm.coef_ * X + lm.intercept_, Y, "Simple Linear Regression based on engine-size", "Model", "Price", "linReg.png")
    ## Multiple linear regression
    Z = auto_data[['engine-size', 'bore', 'highway-mpg']]
    lm.fit(Z, auto_data['price'])
    print("MultReg b = %f" %(lm.intercept_))
    print("MultReg a ", lm.coef_)
    print("MultReg r^2 = %f" % (lm.score(Z, auto_data['price'])))
    model_test(Z @ lm.coef_ + lm.intercept_, Y, "Multiple Linear Regression based on engine-size, bore and highway-mpg", "Model", "Price", "multReg1.png")
    # ## Logarithmic curve fitting
    Z = auto_data[['curb-weight', 'height', 'length']]
    lm.fit(Z, auto_data['price'])
    print("MultReg b = %f" %(lm.intercept_))
    print("MultReg a ", lm.coef_)
    print("MultReg r^2 = %f" % (lm.score(Z, auto_data['price'])))
    model_test(Z @ lm.coef_ + lm.intercept_, Y, "Multiple Linear Regression based on curb-weight, height and length", "Model", "Price", "multReg2.png")

        

