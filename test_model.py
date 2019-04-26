import numpy as np
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from alexnet import alexnet
from getkeys import key_check

import random

WIDTH = 80
HEIGHT = 60
LR = 1e-3
EPOCHS = 5
MODEL_NAME = 'pygta5-{}-{}-{}-epochs-data-1.model'.format(LR, 'alexnet',EPOCHS)

t_time = 0.05

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    time.sleep(t_time)
    ReleaseKey(S)
    ReleaseKey(A)


def right():
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)
    time.sleep(t_time)
    ReleaseKey(S)
    ReleaseKey(D)


def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
    PressKey(A)
    time.sleep(t_time)
    ReleaseKey(A)


def forward_right():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    PressKey(D)
    time.sleep(t_time)
    ReleaseKey(D)
    

def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(D)
    ReleaseKey(W)
    ReleaseKey(A)


def no_keys():
    if random.randrange(0, 3) == 1:
        PressKey(W)
    else:
        ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    
model = alexnet(WIDTH, HEIGHT, LR)
model.load(MODEL_NAME)

def main():
    last_time = time.time()
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        
        if not paused:
            screen = grab_screen(region=(200,200,1000,800))        
##            print('loop took {} seconds'.format(time.time()-last_time))
            last_time = time.time()
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
##            cv2.imshow('boon', screen)
            screen = cv2.resize(screen, (80,60))

            prediction = model.predict([screen.reshape(80,60,1)])[0]
                

            mode_choice = np.argmax(prediction)

            if mode_choice == 0:
                straight()
                choice_picked = 'straight'
            elif mode_choice == 1:
                reverse()
                choice_picked = 'reverse'
            elif mode_choice == 2:
                left()
                choice_picked = 'left'
            elif mode_choice == 3:
                right()
                choice_picked = 'right'
            elif mode_choice == 4:
                forward_left()
                choice_picked = 'forward+left'
            elif mode_choice == 5:
                forward_right()
                choice_picked = 'forward+right'
            elif mode_choice == 6:
                reverse_left()
                choice_picked = 'reverse+left'
            elif mode_choice == 7:
                reverse_right()
                choice_picked = 'reverse+right'
            elif mode_choice == 8:
                no_keys()
                choice_picked = 'nokeys'

            print(choice_picked)

        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
