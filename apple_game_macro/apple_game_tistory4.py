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

from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QPushButton
from PyQt5.QtCore import Qt
class App(QWidget):
    def __init__(self):
        super().__init__()
        self.img_path = './imgs/'
        self.panels = {}
        self.buttons = {}

        self.setWindowTitle('사과게임 매크로')
        self.resize(450, 400)

        self.panel_dict = {
            'BUTTON':{
                'bg-color': 'lightblue',
                'height': 25,
                'text': '',
                'align': QHBoxLayout()
            },
            'INFO_TEXT':{
                'bg-color': 'lightblue',
                'height': 50,
                'text': '',
                'align': QVBoxLayout()
            },
            'STATUS':{
                'bg-color': 'lightblue',
                'height': 300,
                'text': '',
                'align': QVBoxLayout()
            }
        }
        self.setup_panel()

        self.button_dict = {
            'PLAY':{
                'panel_name':'BUTTON',
                'function_name': self.play,
                'width': 100,
                'height':25,
                'idx': 0
            },
            'MACRO':{
                'panel_name':'BUTTON',
                'function_name': self.do_applegame,
                'width': 100,
                'height':25,
                'idx': 1
            },
            'RESET/PLAY':{
                'panel_name':'BUTTON',
                'function_name': self.reset_and_play,
                'width': 100,
                'height':25,
                'idx': 2
            },
            'EXIT':{
                'panel_name':'BUTTON',
                'function_name': self.macro_exit,
                'width': 100,
                'height':25,
                'idx': 3
            },
        }
        self.setup_button()

    def setup_panel(self):
        main_layout = QVBoxLayout()
        for key, value in self.panel_dict.items():
            self.panels[key] = self.create_panel(value)
            main_layout.addWidget(self.panels[key])
        self.setLayout(main_layout)

    def create_panel(self, value):
        panel = QFrame()
        panel.setStyleSheet(f'background-color: {value['bg-color']}')
        panel.setFixedHeight(value['height'])

        layout = value['align']
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel(value['text'])
        
        layout.addWidget(label)
        panel.setLayout(layout)
        
        panel.label = label
        return panel
    
    def setup_button(self):
        for key, value in self.button_dict.items():
            self.button_dict[key] = self.create_button(key, value)

    
    def create_button(self, key, value):
        button = QPushButton(key)
        button.setFixedWidth(value['width'])
        button.setFixedHeight( value['height'])

        button.setStyleSheet(""" 
        QPushButton:pressed {
            background-color: gray;
            color: black;
        }""")

        button.clicked.connect(value['function_name'])
        self.panels[value['panel_name']].layout().insertWidget(value['idx'], button)
        return button


    def get_process(self):
        try:
            self.get_numpy()
            self.get_mouse_pos()
            np_str = '\n'.join(['  '.join(map(str, row)) for row in self.num2d])
            self.set_yhxw()
            self.panels['STATUS'].label.setText(np_str)
            self.panels['STATUS'].label.setAlignment(Qt.AlignCenter)  # 중앙 정렬 (수평 및 수직)
            self.panels['STATUS'].label.setStyleSheet("font-family: Arial; font-size: 20px;")
        except: 
            self.panels['STATUS'].label.setText('문제 발생')
            self.panels['STATUS'].label.setAlignment(Qt.AlignCenter)  # 중앙 정렬 (수평 및 수직)
            self.panels['STATUS'].label.setStyleSheet("font-family: Arial; font-size: 20px;")
            
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
        self.num2d_process = []
        for _ in range(10):
            for yi in range(10):
                for xi in range(17):
                    w_max, h_max = 17 - xi, 10 - yi
                    for h in range(h_max):
                        for w in range(w_max):
                            if np.sum(self.num2d[yi: yi + h+1, xi: xi + w+1]) == 10:
                                self.num2d[yi: yi + h+1, xi: xi + w+1] = 0
                                self.yhxw.append((yi, h, xi, w ))
                                self.num2d_process.append(self.num2d.copy())
        self.score = np.count_nonzero(self.num2d == 0)
        self.panels['INFO_TEXT'].label.setText(f' 예상점수: {self.score}점\n 총 이동횟수   : {len(self.yhxw)}회\n 실제 이동횟수: 0 회')
    

    def do_applegame(self):
        def worker():
            for i, (yi, h, xi, w) in enumerate(self.yhxw):
                if not self.macro_running:
                    self.yhxw = self.yhxw[i:]
                    break
                self.mouse_move(xi, yi, w, h)
                np_str = '\n'.join(['  '.join('_' if x == 0 else str(x) for x in row) for row in self.num2d_process[i]])
                self.panels['INFO_TEXT'].label.setText(f' 예상점수: {self.score}점\n 총 이동횟수   : {len(self.yhxw)}회\n 실제 이동횟수: {i+1}회')
                self.panels['STATUS'].label.setText(np_str)

        self.macro_running = True
        threading.Thread(target=worker).start()

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

    def play(self):
        try:
            self.click_center(f'{self.img_path}play.png')
            self.get_process()
        except:
            self.get_process()

    def reset_and_play(self):
        self.click_center(f'{self.img_path}reset.png')
        self.click_center(f'{self.img_path}play.png')
        self.get_process()

    def click_center(self, img_path):
        positions = pyautogui.locateAllOnScreen(img_path, confidence=0.90)
        for pos in positions:
            x_center = pos.left + pos.width / 2
            y_center = pos.top + pos.height / 2
            pyautogui.moveTo(x_center, y_center, duration=0.01)
            pyautogui.mouseDown()
            pyautogui.mouseUp()
            break
    def macro_exit(self):
        sys.exit()

if __name__ == "__main__":
    app = QApplication([])
    window = App()
    window.show()
    app.exec_()
