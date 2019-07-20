import pygame
#import sys

pygame.init()  # 初始化pygame
size = width, height = 1, 1  # 设置窗口大小
screen = pygame.display.set_mode(size)  # 显示窗口
direction = []
while True:  # 死循环确保窗口一直显示
    for event in pygame.event.get():  # 遍历所有事件
        if event.type == pygame.KEYDOWN and event.key ==pygame.K_RIGHT:
            print("turn right!")
            rb.go_right()
            direction.append(1)

        elif event.type == pygame.KEYDOWN and event.key ==pygame.K_LEFT:
            print("turn left!")
            rb.go_left()
            direction.append(-1)

        else:
            print("go straight")
            rb.one_step(0.01)
            direction.append(0)

        if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出
            sys.exit()

pygame.quit()  # 退出pygame
