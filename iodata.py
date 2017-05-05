import numpy as np

def read_data(file):
    raw_data =np.loadtxt(file)
    data = raw_data[:, 2].reshape((36, 36))
    x = raw_data[:36, 1]
    x = np.append(x, x[-1]*2-x[-2])
    b = np.zeros((data.shape[0]+1, data.shape[1]+1)); b[:-1, :-1] = data
    b[:,-1] = b[:,0]
    b[-1,:] = b[0,:]
    b[-1,-1] = b[0, 0]
    data = b

    y = x
    return x, y, data

def read_data_test(file):
    x = np.linspace(0, 2*np.pi*4, 200)
    y = np.linspace(0, 6*np.pi*4, 200)

    def potential_data(x, y):
        pot = np.sin(x)+np.sin(y)+np.cos(x*0.5)+np.cos(y*0.75)
        return pot

    data = potential_data(*np.meshgrid(x, y, indexing='ij', sparse=True))

    return x, y, data

