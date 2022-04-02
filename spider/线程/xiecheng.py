import asyncio
import time
# 携程：　遇到IO　操作　选择性切换到其他任务上
async def func():
    print("hello")
    # time.sleep(3)
    await asyncio.sleep(3) #挂起异步操作
    print("hello1")

async def func2():
    print("hello")
    await asyncio.sleep(4)
    # time.sleep(4)
    print("hello1")

async def func3():
    print("hello")
    # time.sleep(2)
    await asyncio.sleep(2)
    print("hello1")

async def main():
    # f1 = func()
    # await f1
    # f2 = func2()
    # await f2
    # f3 = func3()
    # await f3
    tasks = [
        asyncio.create_task(func),
        asyncio.create_task(func2),
        asyncio.create_task(func3)
    ]
    await asyncio.wait(tasks)

if __name__ == '__main__':
    asyncio.run(main())

    # futures = [...]
    # loop = asyncio.get_event_loop()
    # loop.run_until_complete(asyncio.wait(futures))

# if __name__ == '__main__':
#     g = func()
#     g1= func2()
#     g2 = func3()
#     tasks=[g,g1,g2]
#     asyncio.run(asyncio.wait(tasks))