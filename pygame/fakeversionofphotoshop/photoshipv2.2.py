from pic import *
class CLS_photoship(object):
    def __init__(self):
        pygame.init()
        self.scr = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        pygame.display.set_caption('RT Photoship')
        self.clock = pygame.time.Clock()
        self.font32 = pygame.font.Font('simkai.ttf',32)
        self.spd=5
        self.mousePos = (0,0)
        self.guideList = []
        self.guideId=0
    def play(self):
        for guide in self.guideList:
            guide.draw(self.scr)
        pygame.display.update()
        self.clock.tick(50)
    def add_guide(self,guide):
        guide.id=len(self.guideList)
        self.guideList.append(guide)
    def keyup(self,key):
        return
    def keydown(self, key):
        return
    def mousedown(self, pos, btn):
        self.guideList[self.guideId].mousedown(pos,btn)
    def mouseup(self, pos, btn):
        self.guideList[self.guideId].mouseup(pos,btn)
    def mousemotion(self, pos):
        self.guideList[self.guideId].mousemotion(pos)
        


class CLS_guide(object):
    def __init__(self, picName):
        self.pic =CLS_pic(picName)
        self.btnList =[]
        self.txtList = []
        self.id = 0
    def draw(self, scr):
        if fwork.guideId != self.id:
            return
        scr.blit(self.pic.img,(0,0))
        for btn in self.btnList:
            btn.draw(scr)
        for txt in self.txtList:
            txt.draw(scr)
    def add_button(self, name ,picFile,x,y,guideId):
        b = CLS_button(name, picFile, x, y, guideId)
        self.btnList.append(b)
    def add_txt(self, txt,font,x,y,c,rect):
        t =CLS_txt(txt,font,x,y,c,rect)
        self.txtList.append(t)
    def mousedown(self,pos,button):
        for btn in self.btnList:
            btn.mousedown(pos, button)
    def mouseup(self,pos,button):
        for btn in self.btnList:
            btn.mouseup(pos, button)
    def mousemotion(self,pos):
        fwork,mousePos = pos


class CLS_button(object):
    def __init__(self, name ,picFile, x,y,guideId):
        self.name=name
        self.img = pygame.image.load(picFile)
        self.img.set_colorkey((38,38,38))
        self.w,self.h = self.img.get_width() // 2, self.img.get_height()
        self.x ,self.y = x, y
        self.rect = pygame.Rect(self.x,self.y,self.w,self.h)
        self.status=0
        self.guideId= guideId
    def draw(self ,scr):
        scr.blit(self.img,(self.x,self.y)),(self.status * self.rect.w,0,\
                                              self.rect.w,self.rect.h)
    def mousedown(self, pos ,button):
        if self.rect.collidepoint(pos):
            self.status=1

    def mouseup(self, pos ,button):
        self.status=0
        if not self.rect.collidepoint(pos):
            return
        if self.name == 'U':
            fwork.guideList[self.guideId].pic.draw(fwork.scr,1,fwork.spd)
        elif self.name == 'D':
            fwork.guideList[self.guideId].pic.draw(fwork.scr,2,fwork.spd)
        elif self.name == 'L':
            fwork.guideList[self.guideId].pic.draw(fwork.scr,3,fwork.spd)
        elif self.name == 'R':
            fwork.guideList[self.guideId].pic.draw(fwork.scr,4,fwork.spd)
        fwork.guideId= self.guideId
        
class CLS_txt(object):
    def __init__(self,txt,font,x,y,c,rect):
        self.txt=txt
        self.img = font.render(txt,True,c)
        self.x,self.y = x,y
        self.c = c
        self.rect= pygame.Rect(rect)
    def draw(self, scr):
        if self.rect.collidepoint(fwork.mousePos):
            scr.blit(self.img,(self.x,self.y))
            
fwork=CLS_photoship()
G01 = CLS_guide('xian01.jpg')
fwork.guideId = G01.id
fwork.add_guide(G01)
G02 = CLS_guide('xian02.jpg')
fwork.add_guide(G02)
G03 = CLS_guide('xian03.jpg')
fwork.add_guide(G03)
G04 = CLS_guide('xian04.jpg')
fwork.add_guide(G04)
G01.add_button('U','bUp.bmp',SCREEN_W //2 -35,20,G02.id)
G01.add_button('L','bLeft.bmp',20, SCREEN_W //2 -35,G03.id)
G01.add_button('R','bRight.bmp',SCREEN_W -100,SCREEN_W //2 -35,G04.id)
G02.add_button('D','bDown.bmp',SCREEN_W //2 -35,SCREEN_H-100,G01.id)
G03.add_button('R','bRight.bmp',SCREEN_W -100,SCREEN_W //2 -35,G01.id)
G04.add_button('L','bLeft.bmp',20, SCREEN_W //2 -35,G01.id)

G01.add_txt('我帅吗？', fwork.font32, 372, 386, (0,255,0), (425,450,20,21))

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
            print(event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            fwork.mouseup(event.pos,event.button)
        elif event.type == pygame.MOUSEMOTION:
            fwork.mousemotion(event.pos)
    fwork.play()
