import pygame, sys , random
SCREEN_WIDTH , SCREEN_HEIGHT = 1025,768
class CLS_disk(object):
    group = []
    def __init__(self,rect,color,speedX, speedY):
        self.rect = pygame.Rect(rect)
        self.color = color
        self.speedX,self.speedY, self.accY = speedX,speedY ,0.02
        CLS_disk.group.append(self)
    def run(self):
        self.speedY += self.accY
        self.rect.x += self.speedX
        self.rect.y += self.speedY
    def draw(self,scr):
        pygame.draw.ellipse(scr,self.color,self.rect,0)
class CLS_gun(object):
    def __init__(self,x,y,r):
        self.x, self.r, self.y = x,r,y
        self.score = 0
        self.diskNum = 10
        self.bulletNum = 0
        self.fireTime = 0
        self.hiScore= 0
    def update(self):
        if pygame.time.get_ticks() - self.fireTime > 100:
            self.fireTime = 0
    def draw(self,scr):
        self.update()
        x,y,r = self.x ,self.y ,self.r
        pygame.draw.circle(scr,(255,255,255),(x,y),r,1)
        pygame.draw.circle(scr,(255,255,255),(x,y),int(r*0.4),1)
        pygame.draw.line(scr,(155,155,155),(x-r,y),(x+r,y),1)
        pygame.draw.line(scr,(155,155,155),(x,y-r),(x,y+r),1)
        if self.fireTime > 0:
            pygame.draw.polygon(scr,(255,0,0),\
                                ((x-int(r*0.4),y-4),(x-int(r*0.4),y+4),(x,y)),0)
            pygame.draw.polygon(scr,(255,0,0),\
                                ((x+int(r*0.4),y-4),(x+int(r*0.4),y+4),(x,y)),0)

def RT_drawb(scr,data,clrList,x0,y0,dw,scale):
    for dy in range(len(data)):
        line = data[dy]
        for dx in range(dw):
            c = clrList[line & 1]
            tx = x0 + (dw - dx -1 )*scale
            ty = y0 + dy * scale
            if scale > 1 :
                pygame.draw.rect(scr,c,(tx,ty,scale,scale),0)
            else:
                scr.set_at((tx,ty),c)
            line = line >> 1
    return
def read_txt(txtname):
    f = open(txtname, 'r')
    txtLine = f.readline()
    f.close()
    dataList= txtLine.split()
    for p in range(len(dataList)):
        dataList[p] = int(dataList[p],16)
    return dataList
def RT_getimg(fn,listname):
    dList = read_txt(fn)
    img = pygame.Surface((64,64))
    RT_drawb(img,dList,listname,0,0,8,8)
    return img
bBrick = [0xff,0x04,0x04,0x04,0xff,0x80,0x80,0x80]
brickClrList = [[64,64,64],[255,127,80]]
bTree = [0x02,0x15,0x07,0x19,0x2e,0x1f,0xfb,0x62]
treeClrList = [[0,50,0],[0,120,0]]
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.Font(None,32)
pygame.mouse.set_visible(False)
gun = CLS_gun(SCREEN_WIDTH //2 ,SCREEN_HEIGHT //2 ,30)
t0= pygame.time.get_ticks()
t1 = random.randint(0,3000)+3000
hiScore = 0

while True :
    for event in pygame.event.get():
        if event.type == pygame.MOUSEBUTTONDOWN and gun.bulletNum>0:
            gun.bulletNum -=1
            i = 0
            gun.fireTime = pygame.time.get_ticks()
            while i < len(CLS_disk.group):
                d = CLS_disk.group[i]
                if d.rect.collidepoint(gun.x,gun.y):
                    CLS_disk.group.pop(i)
                    gun.score+=1
                    if gun.score> gun.hiScore:
                        gun.hiScore= gun.score
                i+=1
                
        if event.type == pygame.KEYDOWN:
            if ord('r')<=event.key <= ord('r'):
                gun.score =0
                gun.diskNum = 10
        if event.type == pygame.MOUSEMOTION:
            gun.x,gun.y = event.pos
        if event.type ==pygame.QUIT:
            pygame.quit()
            sys.exit()
    if pygame.time.get_ticks()-t0 > t1 and gun.diskNum >0:
        gun.diskNum -=1
        gun.bulletNum = 2
        w =random.randint(40,80)
        h = w//2
        disk = CLS_disk((0,SCREEN_HEIGHT,w,h),(0,255,0),\
                        random.random()+1.5,-4.5+random.random())
        t0 = pygame.time.get_ticks()
        t1 = random.randint(0,3000)+3000
        if random.random()<0.3:
            disk = CLS_disk((SCREEN_WIDTH, SCREEN_HEIGHT,w,h),\
                            (255,0,0),-2.5+random.random(),-4.5+random.random())
    
    screen.fill((20,20,144))
    for disk in CLS_disk.group:
        disk.run()
        disk.draw(screen)
    gun.draw(screen)
    img= font.render('SCORE:'+str(gun.score)+'     DISKS:'\
                     + str(gun.diskNum)+'      BULLETS:'+str(gun.bulletNum),True ,(240,0,140))
    screen.blit(img,(0,0))
    img=RT_getimg('brick.txt',brickClrList)
    for x in range(0, SCREEN_WIDTH,64):
        screen.blit(img,(x,700))
    for x in range(0, SCREEN_WIDTH,64):
        RT_drawb(screen,bTree,treeClrList,x,700-64,8,8)
    
    img= font.render('Hi-SCORE:'+str(gun.hiScore),True ,(240,0,140))
    screen.blit(img,(800,0))
    if gun.diskNum == 0 and gun.bulletNum == 0:
        pygame.time.wait(20)
        screen.fill((0,0,0))
        img= font.render('Your SCORE is:'+str(gun.score),True ,(255,255,255))
        screen.blit(img,(SCREEN_WIDTH //2 -50,SCREEN_HEIGHT //2 -100))
        img= font.render('PRESS R TO RESTART',True ,(255,255,255))
        screen.blit(img,(SCREEN_WIDTH //2 -50,SCREEN_HEIGHT //2 -200))
    pygame.display.update()
    clock.tick(300)
