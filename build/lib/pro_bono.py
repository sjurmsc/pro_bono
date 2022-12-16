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
