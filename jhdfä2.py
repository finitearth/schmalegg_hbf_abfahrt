import pyautogui
import cv2
import numpy as np
import time
import keyboard

from PIL import Image

x_click = 1899
y_click = 184
x = 482
y = 160
x2= 1438
y2 = 875
c = 0
image_list = []
try:
    while True:
        pyautogui.click(x_click, y_click)
        time.sleep(1.5)
        image = pyautogui.screenshot()
        image = np.array(image)
        image = image[y:y2, x:x2]
        image = Image.fromarray(image)

        image_list.append(image)

        if keyboard.is_pressed('b'):
            raise KeyboardInterrupt(":)")

except KeyboardInterrupt as _:
    print("jaaaaaaajaaaaaaaaaaaa")
    try:
        input()
    except KeyboardInterrupt:
        input()
    image_list[0].save(r'mckirchy.pdf', save_all=True, append_images=image_list[1:])
