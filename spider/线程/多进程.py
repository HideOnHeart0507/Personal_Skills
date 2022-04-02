
from multiprocessing import Process

def func(a):
    for i in range(1000):
        print(a, i)

if __name__ == '__main__':
    p = Process(target=func, args=("ss",))
    p.start()
    p2 = Process(target=func, args=('aaa',))
    p2.start()