import aiohttp
import asyncio
urls = [
    "https://www.youmeitu.com/Upload/20200629/1593413389315412.jpg"
]

async def aiodownload(url):
    name = url.rsplit("/",1)[1]
    async with aiohttp.ClientSession() as session:#== requests
        async with session.get(url) as resp:
            with open(name, mode='wb') as f:
                f.write(await resp.content.read())


async def main():
    tasks = []
    for url in urls:
        tasks.append(aiodownload(url))
    await asyncio.wait(tasks)

if __name__ == '__main__':
    asyncio.run(main())