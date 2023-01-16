"""

"""
from sklearn.model_selection import train_test_split
import numpy as np
from pro_bono import compiled_LSTM_func
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# Hent inn data her:
gpr = np.load('...')
water_matrix = np.load('...')

gpr_for_prediction = np.load('...')

if __name__ == '__main__':
    N = len(np.unique(water_matrix))
    if N == 2:
        seg_data = {'white' : [(1, 1, 1)], 
                    'blue' : [[0, 0, 1]]}
    else:
        raise ValueError('There should only be two unique labels for watertable')
    cmap = LinearSegmentedColormap('GPR', segmentdata=seg_data, N=N)
    predictor = compiled_LSTM_func(X=gpr, y=water_matrix)
    predicted_water = predictor(gpr_for_prediction)

    plt.imshow(gpr_for_prediction)
    plt.imshow(predicted_water, cmap = cmap, alpha=0.4)
    plt.show()