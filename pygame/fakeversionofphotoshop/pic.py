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
            for w in range(1,SCREEN_W, spd):
                h=int(w*SCREEN_H / SCREEN_W)
                img = pygame.transform.scale(self.img,(w,h))
                scr.blit(img, ((SCREEN_W - w)/2, (SCREEN_H -h)/2))
                pygame.display.update()
        elif effNum ==2:
            oldImg= scr.copy()
            for w in range(0,SCREEN_W,spd):
                h=int(w*SCREEN_H / SCREEN_W)
                img = pygame.transform.scale(oldImg,(w,h))
                scr.blit(img, ((SCREEN_W - w)/2, (SCREEN_H -h)/2))
                pygame.display.update()
        elif effNum == 3 :
            oldImg= scr.copy()
            for x in range(-SCREEN_W, 0, spd):
                scr.blit(self.img,(x,0))
                scr.blit(oldImg,(x+SCREEN_W,0))
                pygame.display.update()
        elif effNum == 4:
            oldImg= scr.copy()
            for x in range(SCREEN_W, 0, -spd):
                scr.blit(self.img,(x,0))
                scr.blit(oldImg,(x-SCREEN_W,0))
                pygame.display.update()
        scr.blit(self.img,(self.x,self.y),(0,0,self.w,self.h))
        pygame.draw.rect(scr,(0,255,0),(0,0,spd*8,8),0)

    def filter(self,scr,filterNum):
        if filterNum == 1:
            for y in range(self.h):
                for x in range(self.w):
                    r0,g0,b0,alpha=self.img.get_at((x,y))
                    gray = int((r0+g0+b0)/3)
                    c = (gray,gray,gray)
                    scr.set_at((x,y),c)
                pygame.display.update()
        elif filterNum ==2:
            img0 = pygame.transform.scale(self.img,(SCREEN_W,SCREEN_W))
            img1 = pygame.Surface((SCREEN_W,SCREEN_W))
            n=self.w
            a= int(self.w/2)
            for y in range(a):
                for x in range(y,a):
                    img1.set_at((x,y),img0.get_at((x,y)))
                    img1.set_at((y,x),img0.get_at((x,y)))
                    img1.set_at((n-1-x,y),img0.get_at((x,y)))
                    img1.set_at((n-1-y,x),img0.get_at((x,y)))
            for y in range(a-1):
                for x in range(n):
                    img1.set_at((x, n-1-y),img1.get_at((x,y)))
            scr= pygame.transform.scale(img1, (SCREEN_W,SCREEN_H))
        self.img =scr.copy()
