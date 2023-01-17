import multiprocessing as mp
import numpy as np


def init(_data):
    global data
    data = _data  # data is now accessible in all children, even on Windows

def task(i):
    return data[i].max() * i



def main():
    smear = np.random.rand(100,100,100)

    pool = mp.Pool( initializer=init, initargs=(smear,))
    print(pool.map(task, range(100)))


    #do multiprocessing while sharing the same variable
    #args = [i for i in range(0, 100)]
    #with mp.Pool( ) as pool:
       # for result in pool.imap_unordered(task, args):
          #  print(result)



if __name__ == '__main__':
    main()