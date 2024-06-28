import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import cv2
import time
from config import *
import os

orig_dir = os.getcwd()
adb_dir = os.path.join(os.getcwd(), "scrcpy-win64-v2.4")

def tap_once(x,y):
	try:
		os.chdir(adb_dir)
		tap_coordinates = 'adb shell input tap '+str(x)+' '+str(y)
		os.system(tap_coordinates) 
		os.chdir(orig_dir)
	except Exception as e:
		print(e)
		input('>> adb failed, press Enter to continue')
		
def hold(x,y,t):
	try:
		os.chdir(adb_dir)
		tap_coordinates = f'adb shell input swipe {str(x)} {str(y)} {str(x)} {str(y)} {str(t)}'
		os.system(tap_coordinates) 
		os.chdir(orig_dir)
	except Exception as e:
		print(e)
		input('>> adb failed, press Enter to continue')

class ScrcpyGameEnv(gym.Env):
    def __init__(self):
        super(ScrcpyGameEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([3, 2, 11])
        self.observation_space = spaces.Box(low=0, high=255, shape=(2336, 1080, 1), dtype=np.uint8)

        command = [SCRCPY_PATH, "--select-usb"]
        self.scrcpy_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def reset(self):
        self.pause_menu('pause')
        self.pause_menu('restart')
        
        observation = self.capture_screen()
        return observation

    def step(self, action):
        self.movement(action)
        time.sleep(0.1 * action[2])  # Wait for the duration of the action

        observation = self.capture_screen()

        reward = self.calculate_reward()

        done, info = self.check_game_over()
        return observation, reward, done, info

    def capture_screen(self):
        os.system('scrcpy-win64-v2.4\\adb.exe exec-out screencap -p > screenshot.png')
        img = cv2.imread('screenshot.png')
        return img

    def movement(self, action):
        actions = {
            'left': (280, 950),
            'right': (555, 950),
            'up': (2100, 868)
        }
        
        move_action, jump_action, duration = action
        duration_ms = duration * 100  # Convert to milliseconds

        if jump_action == 1:
            x, y = actions['up']
            hold(x, y, duration_ms)
        
        if move_action == 1:
            x, y = actions['left']
            hold(x, y, duration_ms)
        elif move_action == 2:
            x, y = actions['right']
            hold(x, y, duration_ms)
    def pause_menu(self, action):
        actions = {
            'pause':(2250, 169),
            'unpause':(2200, 400),
            'restart':(2200, 600)
        }
        
        x,y = actions.get(action, "Invalid")
        tap_once(x,y)
    def calculate_reward(self):
        done, wl = self.check_game_over()
        if done:
            if wl=="win":
                return 5
            else :
                return -3
        else:
            if self.check_movement(): return 1 
            else: return -3
        
    def check_movement(self):
        last_player_position = self.get_player_position()
        time.sleep(0.05)
        current_player_position = self.get_player_position()
        if last_player_position != current_player_position:
            print("Moved")
            return True
        else:
            print("Not moved")
            return False

    def get_player_position(self):
        player_imgs = ['resources\\0.png', 'resources\\1.png', 'resources\\2.png', 
                       'resources\\3.png', 'resources\\4.png', 'resources\\5.png', 
                       'resources\\6.png', 'resources\\7.png', 'resources\\8.png', 
                       'resources\\9.png', 'resources\\10.png', 'resources\\11.png', 
                       'resources\\13.png', 'resources\\14.png', 
                       'resources\\15.png', 'resources\\16.png', 'resources\\17.png', 
                       'resources\\18.png']
        best_score = 0
        best_pos = None
        for img_path in player_imgs:
            player_img = cv2.imread(img_path)
            h, w = player_img.shape[:2]
            res = cv2.matchTemplate(self.capture_screen(), player_img, cv2.TM_CCOEFF_NORMED)
            _, max_val, max_loc, _ = cv2.minMaxLoc(res)
            if max_val > best_score:
                best_score = max_val
                best_pos = (max_loc[0] + w // 2, max_loc[1] + h // 2)
        return best_pos if best_score >= 0.9 else None

    def check_game_over(self):
        def check_win(frame):
            flag_pole_1 = cv2.imread('resources\\flag_1.png')
            flag_pole_2 = cv2.imread('resources\\flag_2.png')
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            flag_pole_1 = cv2.cvtColor(flag_pole_1, cv2.COLOR_BGR2GRAY)
            flag_pole_2 = cv2.cvtColor(flag_pole_2, cv2.COLOR_BGR2GRAY)
            result1 = cv2.matchTemplate(frame, flag_pole_1, cv2.TM_CCOEFF_NORMED)
            result2 = cv2.matchTemplate(frame, flag_pole_2, cv2.TM_CCOEFF_NORMED)

            # threshold compare to match confidence
            threshold = 0.8

            loc1 = cv2.findNonZero((result1 >= threshold).astype(int))
            loc2 = cv2.findNonZero((result2 >= threshold).astype(int))

            if loc1 is not None or loc2 is not None:
                print("Game over")
                return True  
            else:
                print("Not game over")
                return False

        def check_lose():
            gray = cv2.cvtColor(self.capture_screen(), cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            mean_val = cv2.mean(blurred)[0]

            turn_black_threshold = 0.05
            if mean_val < turn_black_threshold:
                return True
            else:
                return False

        if check_win(self.capture_screen()):
            return True,"win"
        elif check_lose():
            return True,"lose"
        else:
            return False,"yet"

    def close(self):
        self.scrcpy_proc.terminate()
