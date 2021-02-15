import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def scatter_plot(attr1, attr2):
    attribute = auto_data[attr1].to_numpy(dtype = float)
    price = auto_data[attr2].to_numpy(dtype = int)
    plt.scatter(attribute, price)
    plt.xlabel(attr1)
    plt.ylabel(attr2)
    plt.title("%s vs %s" % (attr1, attr2))
    plt.show()
if __name__ == '__main__':
    file_path = '~/Documents/GitHub/ECE580/imports-85.data'
    raw_data = pd.read_csv(file_path, delimiter = ',', engine = 'python')
    column_names = ['symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    raw_data.columns = column_names
    ## Preserve only relevant attributes
    important_cols = ['wheel-base', 'length', 'width', 'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 'highway-mpg', 'price']
    global auto_data
    auto_data = raw_data[important_cols]
    ## Remove unknown price instances
    for col in important_cols:
        auto_data = auto_data[auto_data[col] != '?']
    ## Change data type to float for all datapoints
    auto_data.astype('float32').dtypes

    ## Plot each feature
    for attribute in important_cols:
        scatter_plot(attribute, 'price')
        

