import numpy as np

def forcepy_tsi_init( ):

    bandnames = ['mean', 'sd', 'median']

    return bandnames


# pixel function
def forcepy_tsi(args):
    ts, date, nodata = args

    m = np.mean(ts)
    s = np.std(ts)
    d = np.median(ts)

    return (m, s, d)
