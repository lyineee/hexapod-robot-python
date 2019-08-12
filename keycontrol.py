import pygame
import sys
import time


class KeyControl(object):
    def __init__(self):
        self.size = width, height = 5, 5  # 设置窗口大小
        
    def first_run(self):
        pygame.init()  # 初始化pygame

    def start(self):
        screen = pygame.display.set_mode(self.size)  # 显示窗口
        direction = []

    def get_key(self):
        state = None
        for event in pygame.event.get():  # 遍历所有事件
            if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                state = 1
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_d:
                state = 2
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_s:
                state = 3
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_j:
                state = 4
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_k:
                state = 5
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_l:
                state = 6
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_g:
                state = 7 #capture reset
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_h:
                state = 8 # quit
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_n:
                state = 9 # capture end
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_p:
                state = 10 #start capture
            else:
                state = 0
            if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出
                self.end()
        return state

    def end(self):
        print('quit')
        time.sleep(0.8)
        pygame.quit()  # 退出pygame
        print('quit success')
        # sys.exit()


if __name__ == "__main__":
    key = KeyControl()
    key.start()
