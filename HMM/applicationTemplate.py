#!/usr/bin/env python
#_*_coding:utf-8_*_

import HMM
def triangle(length):
    '''
    三角波
    '''
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x = x % 6
        if x <= 3:
            Y.append(x)
        else:
            Y.append(6 - x)
    return X, Y

def sin(length):
    '''
    三角波
    '''
    import math
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x = x % 20
        Y.append(int(math.sin((x * math.pi) / 10) * 50) + 50)
    return X, Y

def show_data(x, y):
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'g')
    plt.show()

    return y

if __name__ == '__main__':
    hmm = HMM.HMM(10, 4)
    tri_x, tri_y = triangle(20)
    hmm.train(tri_y)
    y = hmm.generateQueue(100)
    x = [i for i in range(100)]
    show_data(x, y)

    # hmm = HMM(15,101)
    # sin_x, sin_y = sin(40)
    # show_data(sin_x, sin_y)
    # hmm.train(sin_y)
    # y = hmm.generate(100)
    # x = [i for i in range(100)]
    # show_data(x,y)