import numpy as np
from scipy.special import erfc

def analytical_rate(td):
    '''
    Returns the dimensionless production rate
    Input:
        td: np.array(n)
            Array of dimensionless times
    Returns:
        qd: np.array(n)
            Array of dimensionless production rates
    '''
    pi = np.pi
    term1 = 2*np.exp(-pi**2*td/4.)
    term2 = erfc(1.5*pi*np.sqrt(td))/np.sqrt(pi*td)
    return term1 + term2

if __name__ == '__main__':
    td = np.linspace(0.1, 1.)
    analytical_rate(td)