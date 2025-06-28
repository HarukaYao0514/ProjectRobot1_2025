# Name: Yao Yao
# Student ID: 202283890009
# Course & Year: Project Semester 4
# Date: 28/6/25

from microbit import *
import time

# Define an image of three numbers
num_1 = Image("00900:"
              "09900:"
              "00900:"
              "00900:"
              "09990")

num_2 = Image("09990:"
              "00090:"
              "09990:"
              "09000:"
              "09990")

num_3 = Image("09990:"
              "00090:"
              "09990:"
              "00090:"
              "09990")

# Carousel display
while True:
    display.show(num_1)
    time.sleep(1)
    display.show(num_2)
    time.sleep(1)
    display.show(num_3)
    time.sleep(1)
