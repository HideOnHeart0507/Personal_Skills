from pic import *
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
        self.mousePos = (0,0)
    def load_pic(self):
        fList = ['xian01.jpg','xian02.jpg','xian03.jpg','xian04.jpg']
        for fileName in fList:
            self.picList.append(CLS_pic(fileName))
    def play(self):
        self.picList[self.picCurNum].draw(self.scr,spd=self.spd)
        pygame.draw.circle(self.scr,(0,0,255),self.mousePos,30,3)
        pygame.draw.circle(self.scr,(255,0,0),self.mousePos,15,1)
        pygame.draw.line(self.scr,(255,0,0),(self.mousePos[0]-15,self.mousePos[1]),(self.mousePos[0]+15,self.mousePos[1]),1)
        pygame.draw.line(self.scr,(255,0,0),(self.mousePos[0],self.mousePos[1]-15),(self.mousePos[0],self.mousePos[1]+15),1)
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
        elif ord('1')<= event.key <= ord('9'):
            self.picList[self.picCurNum].filter(self.scr,event.key-ord('0'))
        elif ord('0') == event.key:
            self.picList=[]
            self.load.pic()
    def mousedown(self, pos, btn):
        return
    def mouseup(self, pos, btn):
        return
    def mousemotion(self, pos):
        self.mousePos = pos

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
        elif event.type == pygame.MOUSEBUTTONDOWN:
            fwork.mousedown(event.pos,event.button)
        elif event.type == pygame.MOUSEBUTTONUP:
            fwork.mouseup(event.pos,event.button)
        elif event.type == pygame.MOUSEMOTION:
            fwork.mousemotion(event.pos)
    fwork.play()
