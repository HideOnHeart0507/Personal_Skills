import pygame, time, sys, datetime
SCREEN_WIDTH, SCREEN_HEIGHT = 400,300
ETLED_WIDTH,ETLED_HEIGHT = 40,40
STROKE_WIDTH, STROKE_HEIGHT,DOT_R = 16,12,3
DIGITAL_MASK= [0b00111111,0b00000110,0b01011011,0b01001111,0b01100110,0b01101101,0b01111101,0b00000111,0b01111111,0b01101111]
DIGITAL_MARK= [0x3F,0x06,0x5B,0x4F,0x66,0x6D,0x7D,0x07,0x7F,0x6F]
class CLS_clock(object):
    def __init__(self,x,y):
        self.x,self.y ,self.w,self.h = x,y,ETLED_WIDTH,ETLED_HEIGHT
        self.posList=[(6,2),(21,7),(21,24),(6,36),(2,24),(2,7),(6,19),(29,39)]
        self.status = 0
        self.time = datetime.datetime.now()
        self.num = 0
        self.k = 0
        self.c = c
    def run(self,scr,mark,k):
        if self.status % 4 == 0:
            self.k = 0
            self.time = datetime.datetime.now()
            self.num = self.time.year
            self.draw(scr,mark,k)
        if self.status % 4 == 1:
            self.k = 0
            self.time = datetime.datetime.now()
            self.num = self.time.month * 100 + self.time.day
            self.draw(scr,mark,k)
        if self.status % 4 == 2:
            self.c += 1
            self.time = datetime.datetime.now()
            self.k = self.c % 40
            self.num = self.time.hour*100+ self.time.minute
            self.draw(scr,mark,k)
        if self.status % 4 == 3:
            self.k = 1
            self.time = datetime.datetime.now()
            self.num = self.time.second
            self.draw(scr, mark,k)
        return
    def draw(self,scr,mark,k):
        pygame.draw.rect(scr,(0,0,180),(self.x,self.y,self.w,self.h))
        bit=1
        for i in range(8):#笔段控制位为0 暗色
            c= (0,0,240)
            x0,y0 = self.x+self.posList[i][0],self.y+self.posList[i][1]
            x1,y1 = x0+STROKE_WIDTH, y0+STROKE_HEIGHT
            if mark & bit == bit:
                c=(240,240,240)#笔段控制位为1 亮色
            bit = bit <<1
            if i in (0,3,6):#画横线
                pygame.draw.polygon(scr,c,[(x0,y0+2),(x0+2,y0),\
                                           (x1-2,y0),(x1,y0+2),\
                                           (x1,y0+3),(x1-2,y0+5),\
                                           (x0+2,y0+5),(x0,y0+3)])
            elif i == 7:
                if 0< k <=20 :
                    c=(240,240,240)
                pygame.draw.circle(scr,c,(x0+5,y0-40//3),DOT_R,0)
                pygame.draw.circle(scr,c,(x0+5,y0-80//3),DOT_R,0)
            else:#画竖线
                pygame.draw.polygon(scr,c,[(x0+3,y0),(x0+5,y0+2),\
                                           (x0+5,y1-2),(x0+3,y1),\
                                           (x0+2,y1),(x0,y1-2),\
                                           (x0,y0+2),(x0+2,y0)])
        return
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH,SCREEN_HEIGHT))
pygame.display.set_caption('DIGITEL LED 1.1')
clock=pygame.time.Clock()
pFlag,num,k,c=0,0,0,0
clk0=CLS_clock(100,100)
clk1=CLS_clock(100+ETLED_WIDTH,100)
clk2=CLS_clock(100+ETLED_WIDTH*2,100)
clk3=CLS_clock(100+ETLED_WIDTH*3,100)

tm=datetime.datetime.now()

while True:
    clk0.run(screen,DIGITAL_MASK[clk0.num//1000%10]+pFlag,0)
    clk1.run(screen,DIGITAL_MASK[clk1.num//100%10]+pFlag,clk1.k)
    clk2.run(screen,DIGITAL_MASK[clk2.num//10%10]+pFlag,0)
    clk3.run(screen,DIGITAL_MASK[clk3.num%10]+pFlag,0)
    if clk2.status % 4 == 3:
        clk0.draw(screen,0,0)
        clk1.draw(screen,0,1)
    pygame.display.update()
    clock.tick(50)
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if pygame.K_0 <= event.key <= pygame.K_9:
                num=event.keu-pygame.K_0
            elif event.key == pygame.K_SPACE:
                clk0.status+=1
                clk1.status+=1
                clk2.status+=1
                clk3.status+=1
            elif event.key == pygame.K_PERIOD:
                pFlag = 128-pFlag
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

