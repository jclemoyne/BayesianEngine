import numpy as np

seed = np.random.seed(10012231)


def generate_x_y(nrow, ncol):
    x = np.ones((nrow, ncol + 1))
    n = 0; k = 0
    for j in range(ncol + 1):
        for i in range(nrow):
            rv = np.random.rand()
            n = n + 1
            if rv < 0.5:
                x[i, j] = 0
                k = k + 1
    print('== x_y == ', x.shape, ' n = ', n, ' k = ', k)
    print(x)
    return x[:, : ncol], x[:, [ncol]]


if __name__ == '__main__':
    x, y = generate_x_y(10, 5)
    print('== x === ', x.shape)
    print(x)
    print('== y == ', y.shape)
    print(y)
