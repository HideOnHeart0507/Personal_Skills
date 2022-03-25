import pygame, sys, time
SCREEN_W,SCREEN_H = 800,800
def RT_drawb(scr,data,clrList,x0,y0,dw,scale):
    for dy in range(len(data)):
        b = data[dy]
        for dx in range(dw):
            c = clrList[b & 1]
            tx = x0 + (dw - dx -1 )*scale
            ty = y0 + dy * scale
            if scale > 1 :
                pygame.draw.rect(scr,c,(tx,ty,scale,scale),0)
            else:
                scr.set_at((tx,ty),c)
            b = b >> 1
    return
def RT_block_read(fName):
    try:
        txtFile = open(fName, 'r')
    except:
        return None
    txt = txtFile.readline()
    txtFile.close()
    blockList = txt.split()
    blockDataList = []
    for bData in blockList:
        blockDataList.append(int(bData,16))
    return blockDataList

def RT_block_init(fName,cList, dw ,scale,tm):
    pic = pygame.Surface((dw*scale,dw*scale))
    blockList = RT_block_read(fName)
    if blockList != None:
        RT_drawb(pic,blockList, cList,0,0,dw,scale)
    if tm == True:
        pic.set_colorkey (cList[0])
    return pic
class CLS_maze(object):
    def __init__(self, fn , pic0,pic1,x0,y0,n,scale):
        self.x0,self.y0= x0,y0
        self.n ,self.scale = n, scale
        self.data = [[0 for x in range(n)]for y in range(n)]
        self.block = [0]*n
        self. img = pygame.Surface ((n*scale,n*scale))
        self.read(fn)
        for y in range(n):
            for x in range(n):
                if self.data[y][x]==0:
                    self.img.blit(pic0,(x*scale,y*scale))
                else:
                    self.img.blit(pic1,(x*scale,y*scale))
        print(self.block)
        return
    def draw(self,scr):
        scr.blit(self.img,(self.x0,self.y0))
        return
    def read(self,fName):
        try :
            txtFile = open(fName, 'r')
        except:
            return
        txt = txtFile.readline()
        txtFile.close()
        blockTxt = txt.split()
        for y in range(self.n):
            self.block[y],bit = int(blockTxt[y],16),int(blockTxt[y],16)
            for x in range(self.n):
                self.data[y][self.n-1-x] = bit % 2
                bit = bit // 2
        return
    

SPEED_X = [1,0,-1,0]
SPEED_Y = [0,1,0,-1]
class CLS_pacman(object):
    def __init__(self, n, x, y, flag):
        self.n = n
        self.x,self.y = x,y 
        self.flag = flag
        self.mem = []
        self.mem1 = []
        return
    def test(self,grid,flag):
        x = self.x + SPEED_X[flag]
        y = self.y + SPEED_Y[flag]
        return 0 <= x < self.n and 0 <= y < self.n and grid[y][x]==1
    def move(self,grid):
        if self.test(self.flag,grid) == False :
            self.flag = (self.flag +1)% 4
        self.x += SPEED_X[self.flag]
        self.y += SPEED_Y[self.flag]
        return self.x, self.y
    def rhmove(self, grid):
        for df in ( +1 ,0, -1, +2):
            flag1 = (self.flag + df) % 4
            if self.test(grid,flag1) == False:
                continue
            self.flag= flag1
            self.x += SPEED_X[self.flag]
            self.y += SPEED_Y[self.flag]
            break
        return self.x, self.y
    def hsmove(self,grid):
        canFlag = 0
        for df in (-1 , 0, +1):
            flag1 = (self.flag + df) % 4
            if not self.test(grid,flag1):
                continue
            canFlag +=1
            self.mem.append((self.x,self.y,flag1))
        label = self.mem.pop(-1)
        self.x,self.y,self.flag = label
        self.x += SPEED_X[self.flag]
        self.y += SPEED_Y[self.flag]
        return self.x, self.y
    
    def clmove(self,grid):
        canFlag = 0
        for df in (-1 , 0, +1):
            flag1 = (self.flag + df) % 4
            if not self.test(grid,flag1):
                continue
            canFlag +=1
            self.mem.append((self.x,self.y,flag1))
        label = self.mem.pop(0)
        self.x,self.y,self.flag = label
        self.x += SPEED_X[self.flag]
        self.y += SPEED_Y[self.flag]
        return self.x, self.y
    
    def draw(self,scr,maze,pacpic):
        x0,y0,d= maze.x0 , maze.y0 , maze.scale
        a,b = x0+self.x*d,y0+self.y*d
        aList.append([a,b])
        if [a,b] in aList:
            aList.pop()
        scr.blit(pacpic,(a,b))
        return
pygame.init()
screen = pygame.display.set_mode((SCREEN_W,SCREEN_H))
clock = pygame.time.Clock()
fontScore = pygame.font.Font(None,42)
picPac = RT_block_init('pacman.txt',[(0,0,0),(240,240,0)],8,4,True)
picPac1 = RT_block_init('pacman.txt',[(0,0,0),(255,0,0)],8,4,True)
pacman = CLS_pacman(16,0,0,0)
pic0 = RT_block_init('brick.txt',[(255,127,80),(64,64,64)],8,4,False)
pic1 = RT_block_init('tree.txt',[(0,50,0),(0,160,0)],8,4,False)
maze = CLS_maze('maze01.txt',pic0,pic1,20,100,16,32)
aList =[[20,100]]
while True :
    maze.draw(screen)
    pacman.clmove(maze.data)
    pacman.draw(screen,maze,picPac)
    screen.blit(pic0,(100,20))
    screen.blit(pic1,(150,20))
    screen.blit(picPac,(200,20))
    pygame.display.update()
    clock.tick(5)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
