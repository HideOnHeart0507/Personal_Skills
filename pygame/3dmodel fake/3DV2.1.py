#RT Lesson 正投影3维建模，珠穆朗玛峰
#V2.0 预处理+进度条
import pygame, sys
ORIGIN_ALTITUDE, SCALE = 5700, 0.15     #基准海拔与比例尺
DATA_X_MAX, DATA_Y_MAX = 1401, 1401     #矩形边界
SCREEN_W, SCREEN_H = 1400, 640   # pygame窗口大小
def RT_scr_y( z ): #实际高度z转换为屏幕坐标y
    return SCREEN_H - (z - ORIGIN_ALTITUDE) * SCALE#----- pygame init -----
#----- pygame init -----
pygame.init() 
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H)) #产生窗口对象
pygame.display.set_caption("RT 3D V2.0")
#----- data init -----
datafile = open( 'Qomolangma.asc', 'r' )
strList = datafile.readlines()
datafile.close()
maxList, minList = [0] * DATA_X_MAX, [SCREEN_H] * DATA_X_MAX #遮挡线初始化
dataList = []   #2维list，存放每个点的高度
n = 0#数据条数计数
for dataStr in strList:     #遍历数据文本
    dataStr = dataStr.strip('\n')
    line = []
    for s in dataStr.split():
        line.append(eval(s))
    dataList.append(line)
    n += 1 
    pygame.draw.rect(screen, (0,255,0), (0,250,800,100), 2)
    pygame.draw.rect(screen, (0,255,0), (0,250,int(n/DATA_X_MAX*800),100), 0)
    for event in pygame.event.get(): #事件遍历
        if event.type == pygame.QUIT: #关闭窗口事件
            pygame.quit()
            sys.exit()
    pygame.display.update() #屏幕刷新
screen.fill(( 0, 0, 0 ))
#----- draw North-----
#遮挡线初始化，实际海拔非屏幕坐标max必须最低，min最高，实际是无遮挡
maxList, minList = [0] * DATA_X_MAX, [SCREEN_H/SCALE+ORIGIN_ALTITUDE] * DATA_X_MAX 
for y in range(DATA_Y_MAX):
    x0, z0 = 0, dataList[y][DATA_X_MAX - 1]#连线起点从东开始
    for x in range(DATA_X_MAX):
        z = dataList[y][DATA_X_MAX - 1 - x]#从东向西
        flag = 0 #遮挡线是否改变，改变才要绘制连线
        if z < minList[x]:          #下遮挡线变化
            minList[x], flag = z, 1
        if z > maxList[x]:          #上遮挡线变化
            maxList[x], flag = z, 1
        if flag == 1 and abs(z - z0) < 100: #本次未被遮挡且数据合理，绘制                                      #未被遮挡
            pygame.draw.line(screen, (240,240,240), \
                            (x0, RT_scr_y(z0)), (x, RT_scr_y(z)), 1)
        x0, z0 = x, z #当前坐标作为下一条线的起点
    for event in pygame.event.get(): #事件遍历
        if event.type == pygame.QUIT: #关闭窗口事件
            pygame.quit()
            sys.exit()
    pygame.display.update() #屏幕刷新
while True: #主循环
    for event in pygame.event.get(): #事件遍历
        if event.type == pygame.QUIT: #关闭窗口事件
            pygame.quit()
            sys.exit()
    pygame.display.update() #屏幕刷新
            

