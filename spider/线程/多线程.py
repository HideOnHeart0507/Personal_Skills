from threading import Thread
# 进程是资源单位
# 线程是执行单位
# 启动程序默认主线程
# def func():
#     for i in range(100):
#         print("sasa", i)
#
# if __name__ == '__main__':
#     t = Thread(target=func) #创建线程
#     t.start() #执行该线程 只是状态
#     for i in range(1000):
#         print("main", i)

class myThread(Thread):
    def run(self):
        for i in range(1000):
            print('zi', i)


if __name__ == '__main__':
    t = myThread()
    t.start()
    for i in range(1000):
        print('main', i)