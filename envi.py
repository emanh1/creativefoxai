import gymnasium as gym
from gymnasium import spaces
import numpy as np
import subprocess
import cv2
import time
from config import *
import os
from ultralytics import YOLO
from datetime import datetime
import uiautomator2 as u2

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
        self.device = u2.connect()
        #self.device.app_start('com.shing_hk.creative_fox')
        self.action_space = spaces.MultiDiscrete([3, 2, 10])  # 3 move actions, 2 jump actions, 10 duration steps
        self.observation_space = spaces.Box(low=0, high=255, shape=(2336, 1080, 1), dtype=np.uint8)
        command = [SCRCPY_PATH, "--select-usb"]
        self.scrcpy_proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.model = YOLO('obb/creative_fox.pt')
        self.flag_pole_1 = cv2.imread('resources\\flag_1.png')
        self.flag_pole_2 = cv2.imread('resources\\flag_2.png')
        self.dead_sprites = [cv2.imread('resources\\19.png'), cv2.imread('resources\\20.png')]

    def reset(self):
        self.pause_menu('pause')
        self.pause_menu('restart')
        observation = self.capture_screen()
        return observation

    def step(self, action):
        reward = 0
        before_position = self.get_player_position(self.capture_screen())
        self.movement(action)
        time.sleep(0.1 * action[2])  # Wait for the duration of the action
        observation = self.capture_screen()
        after_position = self.get_player_position(observation)
        if before_position and after_position:
            position_diff = after_position[0] - before_position[0]
            print("Before: ", before_position, "After: ", after_position)
            if position_diff < 0:
                reward -= 0.1
                print("Moved left")
            elif position_diff > 0:
                reward += 1
                print("Moved right")
        reward += self.calculate_reward()
        done, info = self.check_game_over()
        return observation, reward, done, info

    def capture_screen(self):
        command = f"{adb_dir}/adb.exe exec-out screencap -p"
        proc = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
        screen_data = proc.stdout.read()
        screen_data = np.frombuffer(screen_data, np.uint8)
        img = cv2.imdecode(screen_data, cv2.IMREAD_COLOR)
        return img
    
    def movement(self, action):
        move_action, jump_action, duration = action
        duration_ms = [500,600,700,800,900,1000,1100,1200,1300,4000][duration]  # Fixed durations
        actions = {'left': (280, 950), 'right': (555, 950), 'up': (2100, 868)}

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
        actions = {'pause':(2250, 169), 'unpause':(2200, 400), 'restart':(2200, 600)}
        x, y = actions.get(action, "Invalid")
        tap_once(x, y)

    def calculate_reward(self):
        done, wl = self.check_game_over()
        if done:
            if wl == "win":
                return 5
            else:
                return -3
        return 0

    def get_player_position(self, screen):
        results = self.model.predict(np.array(screen))
        best_pos = None
        for result in results:
            boxes = result.boxes.xyxy
            class_ids = result.boxes.cls
            for box, class_id in zip(boxes, class_ids):
                class_id = int(class_id)
                if class_id == 1:
                    x1, y1, x2, y2 = box.tolist()
                    best_pos = (int((x1 + x2) // 2), int((y1 + y2) // 2))
        return best_pos

    def check_game_over(self):
        def check_win():
            frame = self.capture_screen()
            result1 = cv2.matchTemplate(frame, self.flag_pole_1, cv2.TM_CCOEFF_NORMED)
            result2 = cv2.matchTemplate(frame, self.flag_pole_2, cv2.TM_CCOEFF_NORMED)

            # threshold compare to match confidence
            threshold = 0.8
            loc1 = cv2.findNonZero((result1 >= threshold).astype(int))
            loc2 = cv2.findNonZero((result2 >= threshold).astype(int))

            if loc1 is not None or loc2 is not None:
                print("Game over - Win")
                if loc1 is not None:
                    (x, y, w, h) = cv2.boundingRect(loc1)
                else:
                    (x, y, w, h) = cv2.boundingRect(loc2)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"win_screenshot_{timestamp}.png", frame)
                return True
            else:
                print("Not game over")
                return False

        def check_lose():
            dead_sprites = self.dead_sprites
            frame = self.capture_screen()
            for sprite in dead_sprites:
                result = cv2.matchTemplate(frame, sprite, cv2.TM_CCOEFF_NORMED)
                threshold = 0.8
                loc = cv2.findNonZero((result >= threshold).astype(int))

                if loc is not None:
                    print("Game over - Lose (Character Dead)")
                    (x, y, w, h) = cv2.boundingRect(loc)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red rectangle
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"lose_screenshot_{timestamp}.png", frame)
                    return True

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            mean_val = cv2.mean(blurred)[0]

            turn_black_threshold = 5
            if mean_val < turn_black_threshold:
                print("Game over - Lose (Black Screen)")
                cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 255), 2)  # Red border for full screen
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"black_screen_{timestamp}.png", frame)
                return True

            return False

        if check_win():
            return True, "win"
        elif check_lose():
            return True, "lose"
        else:
            return False, "yet"

    def close(self):
        self.scrcpy_proc.terminate()
