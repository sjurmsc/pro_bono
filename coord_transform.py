from pyproj import proj
import pandas as pd
import sys
from pathlib import Path

cwd = Path.cwd()

for file in cwd.glob('*.csv'):

    table = pd.read_csv(file, sep=';')

    X = table['X']; Y = table['Y']

    crs_from = 'ETRS89 / UTM zone 32N'
    crs_to   = 'ETRS89 / UTM zone 33N'

    Trans = proj.Transformer.from_crs(crs_from, crs_to)

    nu_X, nu_Y = Trans.transform(X, Y)

    table['X'] = nu_X; table['Y'] = nu_Y

    table.to_csv('{}_33N.csv'.format(str(file)[:-4]), sep=';')
