# -*- coding: utf-8 -*-
import math


"""
Locomotion-related constants
"""
TIME_FRAME = 20 #the length of the simulation step and real robot control frame [ms]

SERVOS_BASE = [	0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.,
                0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.,
                0, 0, -math.pi/6., math.pi/6., math.pi/3., -math.pi/3.]

#robot construction constants
COXA_SERVOS = [1,13,7,2,14,8]
COXA_OFFSETS = [math.pi/8,0,-math.pi/8,-math.pi/8,0,math.pi/8]
FEMUR_SERVOS = [3,15,9,4,16,10]
FEMUR_OFFSETS = [-math.pi/6,-math.pi/6,-math.pi/6,math.pi/6,math.pi/6,math.pi/6]
TIBIA_SERVOS = [5,17,11,6,18,12]
TIBIA_OFFSETS = [math.pi/3,math.pi/3,math.pi/3,-math.pi/3,-math.pi/3,-math.pi/3]
SIGN_SERVOS = [-1,-1,-1,1,1,1]

FEMUR_MAX = 2.1
TIBIA_MAX = 0.1
COXA_MAX = 0.01



"""
Navigation related constants
"""

# DISTANCE_THLD = 0.1 #m
# LOOSE_DISTANCE_THLD = 0.2
# C_TURNING_SPEED = 0.5/math.pi
# BASE_SPEED = 0.5
# DEGREE_THLD = 0.1
# DEGREE_RANGE = 0.3
# LOOSE_DEGREE_THLD = 0.2
DISTANCE_THLD = 0.1 #m
C_TURNING_SPEED = 0.3/math.pi
C_TURNING_SPEED_DOWNSCALE = 0.7
BASE_SPEED = 0.7
DEGREE_THLD = 0.1
SPARE_DIST_MIN = BASE_SPEED/(C_TURNING_SPEED)
W_GOAL = 1
W_DIR = 50*W_GOAL