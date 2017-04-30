#! /usr/bin/env python

"""
Hard Coded AI capable of playing the Atari game Pong.
"""

from __future__ import print_function

__author__ = "Padraig O Neill"




import numpy as np
from PIL import Image
from ale_python_interface import ALEInterface

np.set_printoptions(threshold=np.nan)

rom = "../roms/pong.bin"
ale = ALEInterface()

# Can set a frame skip to make game progress faster, however AI will get worse.
ale.setInt("frame_skip", 1)
max_frames_per_episode = ale.getInt("max_num_frames_per_episode")

ale.setInt("random_seed", 123)
ale.setBool('display_screen', True)
ale.setBool('color_averaging', True)
ale.setString('record_screen_dir', "frames")
ale.loadROM(rom)

total_reward = 0
k = 0

# Keep track of height for player and ball
ball_y = 0
player_y = 0

(screen_width, screen_height) = ale.getScreenDims()
print("width/height: " + str(screen_width) + "/" + str(screen_height))

# If DEBUG_MODE is TRUE, game will pause at every frame and show the region, and print out debug info
DEBUG_MODE = False

# Get the list of legal actions
legal_actions = ale.getLegalActionSet()
if DEBUG_MODE:
    print("legal actions" + str(legal_actions))


def move_up():
    """
    :returns: return int value 3, action for up
    """
    return 3


def move_down():
    """
    :returns: return int value 3, action for down
    """
    return 4


def do_nothing():
    """
    :returns: return int value , action for doing nothing
    """
    return 0


def find_pixels(pixels, red, green, blue, x_value, y_value):
    """
    Method that checks the current pixel, for colours matching either the paddle or the ball

    :param pixels: Array of pixels in region
    :param red: r value
    :param green: g value
    :param blue: b value
    :param x_value: pixel x value
    :param y_value: pixel y value
    :returns: True if pixel is found, other false
    :raises indexError: raises an exception if checks pixel that doesn't exist
    """
    if pixels[x_value, y_value] == (red, green, blue):
        err = ''
        try:
            if pixels[x_value - 1, y_value - 1] != (red, green, blue):
                if DEBUG_MODE:
                    p[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'
        try:
            if pixels[x_value + 1, y_value - 1] != (red, green, blue):
                if DEBUG_MODE:
                    pixels[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'
        try:
            if pixels[x_value - 1, y_value + 1] != (red, green, blue):
                if DEBUG_MODE:
                    pixels[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'

        try:
            if pixels[x_value + 1, y_value + 1] != (red, green, blue):
                if DEBUG_MODE:
                    pixels[x_value - 1, y_value - 1] = (220, 34, 30)
        except IndexError:
            err = 'Index out of range'

        print(err)

        return True
    return False


A = np.zeros(screen_width * screen_height, dtype=np.uint32)

while not ale.game_over():

    ale.saveScreenPNG("frames/frame-%07d.png" % k)
    img_rgb = Image.open("frames/frame-%07d.png" % k)

    box = (0, 34, 320, 194)

    # Crop out unnecessary objects from frame.
    region = img_rgb.crop(box)

    action = 0
    p = region.load()
    width, height = region.size

    if DEBUG_MODE:
        print("Ball Y : " + str(ball_y))
        print("Player Y : " + str(player_y))

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            if find_pixels(p, 92, 186, 92, x, y):
                player_y = y

            if find_pixels(p, 236, 236, 236, x, y):
                ball_y = y

    if DEBUG_MODE:
        region.show()
        img_rgb.paste(region, box)
        img_rgb.show()
        raw_input("Press Enter to continue...")

    # Decide action based on height of ball in relation to player

    if (ball_y - player_y) > -3:
        if DEBUG_MODE:
            print(" Moving Down")
        action = move_down()

    elif (ball_y - player_y) < -12:
        if DEBUG_MODE:
            print(" Moving Up")
        action = move_up()

    else:
        action = do_nothing()
        if DEBUG_MODE:
            print("No info")

    reward = ale.act(action)
    k += 1
    total_reward += reward
    if DEBUG_MODE:
        print(ale.lives(), total_reward)
    screen_data = np.zeros(screen_width * screen_height, dtype=np.uint32)
    ale.getScreenRGB(screen_data)

print("Episode ended with score:", total_reward)
