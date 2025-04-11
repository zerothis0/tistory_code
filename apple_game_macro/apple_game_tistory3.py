import keyboard
import pyautogui
import pandas as pd
import numpy as np
import sys
import time
import threading
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
os.environ["OMP_NUM_THREADS"] = "2"

class App:
    def __init__(self):
        print('d: 사과게임 수행/재개')
        print('f: 사과게임 중단')
        print('q: 종료')
        print('r: Reset 버튼 클릭 + Start 버튼 클릭') 
        self.img_path = './imgs/'
        self.program_running = True
        self.macro_running = True
        play_positions = pyautogui.locateAllOnScreen(f'{self.img_path}play.png', confidence=0.90)
        try:
            if  list(play_positions):
                print('게임을 시작하려면 r을 클릭')
        except:
            self.process()

        keyboard.add_hotkey('d', self.do_applegame)
        keyboard.add_hotkey('f', self.exit_applegame)
        keyboard.add_hotkey('q', self.exit)
        keyboard.add_hotkey('r', self.reset_and_process)

        while self.program_running:
            time.sleep(1.)
    
    def process(self):
        self.get_numpy()
        self.get_mouse_pos()
        self.set_yhxw()

    def do_applegame(self):
        print('d를 누름')
        def worker():
            for i, (yi, h, xi, w) in enumerate(self.yhxw):
                if not self.macro_running:
                    print('사과게임 중단, 재개는 "d"')
                    self.yhxw = self.yhxw[i:]
                    break
                self.mouse_move(xi, yi, w, h)
        self.macro_running = True
        threading.Thread(target=worker).start()
    
    def exit_applegame(self):
        print('f를 누름')
        self.macro_running = False

    def reset_and_process(self):
        print('r을 누름')
        self.click_center(f'{self.img_path}reset.png')
        self.click_center(f'{self.img_path}play.png')
        self.process()

    def click_center(self, img_path):
        positions = pyautogui.locateAllOnScreen(img_path, confidence=0.90)
        for pos in positions:
            x_center = pos.left + pos.width / 2
            y_center = pos.top + pos.height / 2
            pyautogui.moveTo(x_center, y_center, duration=0.01)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            break

    def exit(self):
        print('q를 누름')
        print('프로그램 종료')
        self.program_running = False

    def get_numpy(self):
        ll = 0
        df = pd.DataFrame(columns=['num', 'top', 'left', 'width', 'height'])
        for i in range(1, 9+1):
            positions = pyautogui.locateAllOnScreen(f'{self.img_path}{i}.png', confidence=0.90)
            for pos in positions:
                new_row = {'num':i, 'top':pos.top, 'left':pos.left, 
                        'width':pos.width, 'height':pos.height}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                ll += 1

        kmeans = KMeans(n_clusters=170)
        kmeans.fit(df[['top', 'left']])
        df['cluster'] = kmeans.labels_
        df = df.groupby(by='cluster').mean()

        kmeans = KMeans(n_clusters=17)
        kmeans.fit(df[['left']])
        df['x_cls'] = kmeans.labels_

        kmeans = KMeans(n_clusters=10)
        kmeans.fit(df[['top']])
        df['y_cls'] = kmeans.labels_

        for xi in range(17):
            mean_value = df.loc[df['x_cls'] == xi, 'left'].mean()  # 먼저 평균값 계산
            df.loc[df['x_cls'] == xi, 'left'] = mean_value  # loc를 사용하여 값 업데이트
        for xi in range(10):
            mean_value = df.loc[df['y_cls'] == xi, 'top'].mean()  # 먼저 평균값 계산
            df.loc[df['y_cls'] == xi, 'top'] = mean_value  # loc를 사용하여 값 업데이트

        df = df.sort_values(by=['top', 'left']).reset_index(drop=True)
        self.df = df
        num1d = np.array(df['num'].values.tolist(), dtype=int)
        self.num2d = num1d.reshape(10, 17)

    def get_mouse_pos(self):
        xloc = self.df['left'].unique()
        yloc = self.df['top'].unique()

        self.dx = np.mean(np.diff(xloc))
        self.dy = np.mean(np.diff(yloc))
        self.xloc_mouse = xloc - 0.25 * self.dx
        self.yloc_mouse = yloc - 0.25 * self.dy
    
    def set_yhxw(self):
        self.yhxw = []
        for _ in range(10):
            for yi in range(10):
                for xi in range(17):
                    w_max, h_max = 17 - xi, 10 - yi
                    for h in range(h_max):
                        for w in range(w_max):
                            if np.sum(self.num2d[yi: yi + h+1, xi: xi + w+1]) == 10:
                                self.num2d[yi: yi + h+1, xi: xi + w+1] = 0
                                self.yhxw.append((yi, h, xi, w ))
        self.score = np.count_nonzero(self.num2d == 0)
        print(f'예상점수: {self.score}')
    

    def mouse_move(self, xi, yi, w, h, duration=0.01):
        xloc_mouse = self.xloc_mouse
        yloc_mouse = self.yloc_mouse
        dx = self.dx
        dy = self.dy
        pyautogui.moveTo( xloc_mouse[xi], yloc_mouse[yi], duration=duration)
        pyautogui.mouseDown()
        pyautogui.moveTo(xloc_mouse[xi] + (w+1)*dx, yloc_mouse[yi] + (h+1)*dy,  duration=duration)
        pyautogui.moveTo(xloc_mouse[xi] + (w+1)*dx, yloc_mouse[yi] + (h+1)*dy,  duration=duration)
        pyautogui.mouseUp()


if __name__ == '__main__':
    app = App()
