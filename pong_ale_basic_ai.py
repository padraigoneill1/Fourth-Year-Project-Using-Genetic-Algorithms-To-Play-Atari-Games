#! /usr/bin/env python

from __future__ import print_function

import numpy as np
from PIL import Image
from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

rom = "../roms/pong.bin"  # if len(sys.argv) < 2 else sys.argv[1]
ale = ALEInterface()
max_frames_per_episode = ale.getInt("max_num_frames_per_episode")

ale.setInt("random_seed", 123)
ale.setBool('display_screen', True)
ale.setBool('color_averaging', True)
ale.setString('record_screen_dir', "frames")
ale.loadROM(rom)

total_reward = 0
k = 0
ball_y = 0
player_y = 0

(screen_width, screen_height) = ale.getScreenDims()
print("width/height: " + str(screen_width) + "/" + str(screen_height))

DEBUG_MODE = False

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
if DEBUG_MODE:
    print("legal actions" +  str(legal_actions))


def move_up():
    return 3


def move_down():
    return 4


def do_nothing():
    return 0


def find_pixels(pixel, red, green, blue, x_value, y_value, object_y_value):
    if pixel[x_value, y_value] == (red, green, blue):
        try:
            if pixel[x_value - 1, y_value - 1] != (red, green, blue):
                p[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'
        try:
            if pixel[x_value + 1, y_value - 1] != (red, green, blue):
                pixel[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'
        try:
            if pixel[x_value - 1, y_value + 1] != (red, green, blue):
                pixel[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'

        try:
            if pixel[x_value + 1, y_value + 1] != (red, green, blue):
                pixel[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'

        return True
    return False


A = np.zeros(screen_width * screen_height, dtype=np.uint32)

while not ale.game_over():

    ale.saveScreenPNG("frames/frame-%07d.png" % k)

    img_rgb = Image.open("frames/frame-%07d.png" % k)

    box = (0, 34, 320, 194)
    region = img_rgb.crop(box)

    a = 0
    p = region.load()
    w, h = region.size

    if DEBUG_MODE:
        print ("Ball Y : " + str(ball_y))
        print ("Player Y : " + str(player_y))

    for x in range(0, w, 1):
        for y in range(0, h, 1):
            if find_pixels(p, 92, 186, 92, x, y, player_y):
                player_y = y

            if find_pixels(p, 236, 236, 236, x, y, ball_y):
                ball_y = y

    if DEBUG_MODE:
        region.show()
        img_rgb.paste(region,box)
        img_rgb.show()
        raw_input("Press Enter to continue...")

    if (ball_y - player_y) > -3:
        if DEBUG_MODE:
            print(" Moving Down")
        a = move_down()

    elif (ball_y - player_y) < -12:
        if DEBUG_MODE:
            print (" Moving Up")
        a = move_up()

    else:
        a = do_nothing()
        if DEBUG_MODE:
            print ("No info")


    reward = ale.act(a)
    k += 1
    total_reward += reward
    if DEBUG_MODE:
        print (ale.lives(), total_reward)
    screen_data = np.zeros(screen_width * screen_height, dtype=np.uint32)

    ale.getScreenRGB(screen_data)

print ("Episode ended with score:", total_reward)
# f.close()  # you can omit in most cases as the destructor will call it
