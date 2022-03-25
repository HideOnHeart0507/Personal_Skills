import pygame,sys,random
SCREEN_W, SCREEN_H = 1024,768

class CLS_pic(object):
    def __init__(self,fileName):
        img=pygame.image.load(fileName)
        self.img =pygame.transform.scale(img,(SCREEN_W, SCREEN_H))
        self.x,self.y =0,0
        self.w,self.h = self.img.get_size()
    def draw(self,scr , effNum =0, spd=5):
        
        if effNum ==1:
            for x in range(-SCREEN_W, 0, spd):
                scr.blit(self.img,(x,0))
                pygame.display.update()
        elif effNum ==2:
            for x in range(0,SCREEN_W,spd):
                scr.blit(self.img,(x,0),(x,0,spd,self.h))
                pygame.display.update()
        elif effNum == 3 :
            oldImg= scr.copy()
            for x in range(-SCREEN_W, 0, spd):
                scr.blit(self.img,(x,0))
                scr.blit(oldImg,(x+SCREEN_W,0))
                pygame.display.update()
        scr.blit(self.img,(self.x,self.y),(0,0,self.w,self.h))
        pygame.draw.rect(scr,(0,255,0),(0,0,spd*8,8),0)
class CLS_photoship(object):
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption('RT Photoship')
        self.clock = pygame.time.Clock()
        self.font32 = pygame.font.Font('simkai.ttf',32)
        self.picList = []
        self.load_pic()
        self.picCurNum = 0
        self.spd=5
        self.effNum=0
    def load_pic(self):
        fList = ['xian01.jpg','xian02.jpg','xian03.jpg','xian04.jpg']
        for fileName in fList:
            self.picList.append(CLS_pic(fileName))
    def play(self):
        self.picList[self.picCurNum].draw(self.scr,spd=self.spd)
        pygame.display.update()
        self.clock.tick(50)
    def keyup(self,key):
        return
    def keydown(self, key):
        if event.key in (32 , pygame.K_RIGHT):
            self.picCurNum = (self.picCurNum + 1)% len(self.picList)
            self.picList[self.picCurNum].draw(self.scr,self.effNum,self.spd)
        elif event.key == pygame.K_LEFT:
            self.picCurNum = (self.picCurNum - 1)% len(self.picList)
            self.picList[self.picCurNum].draw(self.scr,self.effNum,self.spd)
        elif event.key == pygame.K_UP:
            self.spd = (self.spd+1)%10
        elif event.key == pygame.K_DOWN:
            self.spd = (self.spd-1)%10
        elif ord('a')<=event.key <= ord('z'):
            self.effNum = event.key - ord('a')+1
            self.picCurNum = (self.picCurNum+1) % len(self.picList)
            self.picList[self.picCurNum].draw(self.scr,self.effNum, self.spd)

fwork=CLS_photoship()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            fwork.keydown(event.key)
        elif event.type == pygame.KEYUP:
            fwork.keyup(event.key)
    fwork.play()
