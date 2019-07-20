import pygame
import sys


class KeyControl(object):
    def __init__(self):
        pygame.init()  # 初始化pygame
        self.size = width, height = 5, 5  # 设置窗口大小

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
            else:
                state = 0
            if event.type == pygame.QUIT:  # 如果单击关闭窗口，则退出
                self.end()
        return state

    def end(self):
        pygame.quit()  # 退出pygame
        sys.exit()


if __name__ == "__main__":
    key = KeyControl()
    key.start()
