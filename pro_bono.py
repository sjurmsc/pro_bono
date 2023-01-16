from pyproj import proj
import pandas as pd
import sys
from pathlib import Path

def convert_crs_for_csv_in_dir():
    cwd = Path.cwd()
    crs_from = 'ETRS89 / UTM zone 32N'
    crs_to   = 'ETRS89 / UTM zone 33N'

    if len(sys.argv) > 2:
        assert len(sys.argv) == 4
        crs_from = str(sys.argv[2]) ; crs_to = str(sys.argv[3])

    try: Trans = proj.Transformer.from_crs(crs_from, crs_to)
    except Exception as e: print(e)

    file_list = [i for i in cwd.glob('*.csv')]

    for file in file_list:
        new_suffix = crs_to.split(' ')[-1] + '.csv'
        filename = '{}_{}'.format(str(file)[:-4], new_suffix)

        if new_suffix in file.name.split('_'): continue  # Makes sure not to redo old files

        table = pd.read_csv(file, sep=';')

        try: X = table['X']; Y = table['Y']
        except: 'File header labels must have X, and Y'

        nu_X, nu_Y = Trans.transform(X, Y)

        table['X'] = nu_X; table['Y'] = nu_Y

        table.to_csv(filename, sep=';', index=False)


import numpy as np
from scipy.interpolate import interp1d
def depth_to_time(ai, z, v, t0 = 0., dt=1e-3, mode='ceil'):
    """
    time units: seconds
    """
    assert np.shape(ai) == np.shape(v)

    if mode == 'ceil':
        v = v[:, :-1]
    elif mode == 'floor':
        v = v[:, 1:]
    
    dz = np.diff(z)
    dt_matrix = 1/v * dz.T

    t = np.zeros_like(ai)
    t[:, 0] = np.ones_like(t[:, 0])*t0

    for i in range(dt_matrix.shape[1]):
        t[:, i+1] = t[:, i] + dt_matrix[:, i]
    
    t_end = np.amin(t[:, -1])
    n_samples = (t_end - t0)/dt
    new_t = np.linspace(t0, t_end, z.shape[0])
    ai_t = np.zeros_like(ai)
    for i in range(ai.shape[0]):
        f = interp1d(t[i], ai[i])
        ai_t[i, :] = f(new_t)
    
    return ai_t


def time_to_depth(gpr, t, x0, v, mode='ceil'):
    assert np.shape(v) == np.shape(gpr)

    if mode == 'ceil':
        v = v[:, :-1]
    elif mode == 'floor':
        v = v[:, 1:]

    xmin = x0 ; xmax = None
    dt = np.diff(t)
    dx = v*dt
    x = np.zeros_like(t)
    x[:, 0] = np.ones_like(x[:, 0])*x0
    for i in range(dx.shape[1]):
        x[:, i+1] = x[:, i] + dx[:, i]

    xmax = np.amin(x[:, -1]) # Setter høyeste x verdi

    new_x = np.linspace(xmin, xmax, np.shape(t)[0])
    new_gpr = np.zeros_like(gpr)
    for i in range(gpr.shape[0]):
        f = interp1d(x[i], gpr[i])
        new_gpr[i, :] = f(new_x)
    
    return new_gpr


from tensorflow import keras
from keras.layers import LSTM, Input
from keras import Model

def compiled_LSTM_func(X, y):
    
    input_layer = Input(X.shape[1:])

    x = LSTM(
        units = 1,
        activation = 'tanh',
        recurrent_activation = 'sigmoid',
        return_sequences = True,

    )(input_layer)
    
    model = Model(inputs=input_layer, outputs=x)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.1), loss=keras.losses.BinaryCrossentropy())
    model.summary()
    model.fit(X=X, y=y)

    return model.predict