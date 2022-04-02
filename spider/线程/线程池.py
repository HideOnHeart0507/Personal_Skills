#一次性开辟一些线程 给予线程池子提交任务
# 线程任务的调度交给线程池完成

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def fn(name):
        for i in range(1000):
            print(name,i)

if __name__ == '__main__':
    with ThreadPoolExecutor(50) as t:
        for i in range(100):
            t.submit(fn, name=f'线程{i}')
    # 守护 ↑  等待线程池中所有任务完成 才会继续执行
    print(123)