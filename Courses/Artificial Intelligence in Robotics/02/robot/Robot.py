#!/usr/bin/env python3
# -*- coding: utf-8 -*-
 
import math
import time
import numpy as np
import threading as thread
 
#hexapod robot 
import hexapod_sim.RobotHAL as robothal
#import hexapod_real.RobotHAL as robothal
 
#cpg network
import cpg.oscilator_network as osc
 
from RobotConst import *
 
class Robot:
    def __init__(self):
        self.robot = robothal.RobotHAL(TIME_FRAME)
        self.v_left = 1
        self.v_right = 1
        self.stop = False
        self.steering_lock = thread.Lock()      #mutex for access to turn commands
 
    def get_pose(self):
        """
        Method to get the robot pose
 
        Returns
        -------
        np.array (float, float, float)
            position and orientation
        """
        coord = self.robot.get_robot_position()
        while coord == None:
            coord = self.robot.get_robot_position()
         
        phi = self.robot.get_robot_orientation()
        while phi == None:
            phi = self.robot.get_robot_orientation()
         
        return np.append(coord, phi)
 
    def turn_on(self):
        """
        Method to drive the robot into the default position
        """
        #read out the current pose of the robot
        pose = self.robot.get_all_servo_position()
 
        #interpolate to the default position
        INTERPOLATION_TIME = 3000 #ms
        interpolation_steps = int(INTERPOLATION_TIME/TIME_FRAME)
 
        speed = np.zeros(18)
        for i in range(0,18):
            speed[i] = (SERVOS_BASE[i]-pose[i])/interpolation_steps
         
        #execute the motion
        for t in range(0, interpolation_steps):
            self.robot.set_all_servo_position(pose + t*speed)
            pass
 
    def locomotion(self):
        """
        method for locomotion control of the hexapod robot
        """
        #cpg network instantiation
        cpg = osc.OscilatorNetwork(6)
        cpg.change_gait(osc.TRIPOD_GAIT_WEIGHTS)
         
        coxa_angles= [0, 0, 0, 0, 0, 0]
        cycle_length = [0, 0, 0, 0, 0, 0]
         
        #main locomotion control loop
        while not self.stop:
            # steering - acquire left and right steering speeds
            self.steering_lock.acquire()
            left = np.min([1, np.max([-1, self.v_left])])
            right = np.min([1, np.max([-1, self.v_right])])
            self.steering_lock.release()
 
            coxa_dir = [left, left, left, right, right, right]      #set directions for individual legs
 
            #next step of CPG
            cycle = cpg.oscilate_all_CPGs()
            #read the state of the network
            data = cpg.get_last_values()
 
            #reset coxa angles if new cycle is detected
            for i in range(0, 6):
                cycle_length[i] += 1;
                if cycle[i] == True:
                    coxa_angles[i]= -((cycle_length[i]-2)/2)*COXA_MAX
                    cycle_length[i] = 0
 
            angles = np.zeros(18)
            #calculate individual joint angles for each of six legs
            for i in range(0, 6):
                femur_val = FEMUR_MAX*data[i] #calculation of femur angle
                if femur_val < 0:
                    coxa_angles[i] -= coxa_dir[i]*COXA_MAX  #calculation of coxa angle -> stride phase
                    femur_val *= 0.025
                else:
                    coxa_angles[i] += coxa_dir[i]*COXA_MAX  #calculation of coxa angle -> stance phase
                 
                coxa_val = coxa_angles[i]
                tibia_val = -TIBIA_MAX*data[i]       #calculation of tibia angle
                     
                #set position of each servo
                angles[COXA_SERVOS[i] - 1] = SIGN_SERVOS[i]*coxa_val + COXA_OFFSETS[i]
                angles[FEMUR_SERVOS[i] - 1] = SIGN_SERVOS[i]*femur_val + FEMUR_OFFSETS[i]
                angles[TIBIA_SERVOS[i] - 1] = SIGN_SERVOS[i]*tibia_val + TIBIA_OFFSETS[i]
                     
            #set all servos simultaneously
            self.robot.set_all_servo_position(angles)
 
    def normalize_angle(self, theta):
        if np.abs(theta) < math.pi:
            return theta
 
        return theta + 2*math.pi if theta < 0 else theta - 2*math.pi
 
    def rotate_to_target(self, coord):
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        theta = np.arctan2(d[1], d[0])
        dphi = self.normalize_angle(theta - current_pose[2])
        while np.abs(dphi) > DEGREE_THLD:
            if self.robot.get_robot_collision():
                return False
 
            self.steering_lock.acquire()
            if dphi > 0:
                self.v_left = -1
                self.v_right = 1
            else:
                self.v_left = 1
                self.v_right = -1
            self.steering_lock.release()
            time.sleep(0.1)
 
            current_pose = self.get_pose()
            d = coord - current_pose[:2]
            theta = np.arctan2(d[1], d[0])
            dphi = self.normalize_angle(theta - current_pose[2])
            # print(dphi, self.v_left, self.v_right)
        return True
 
    def rotate_to_direction(self, phi):
        current_pose = self.get_pose()
        dphi = self.normalize_angle(phi - current_pose[2])
        while np.abs(dphi) > DEGREE_THLD:
            if self.robot.get_robot_collision():
                return False
 
            self.steering_lock.acquire()
            if dphi > 0:
                self.v_left = -1
                self.v_right = 1
            else:
                self.v_left = 1
                self.v_right = -1
            self.steering_lock.release()
            time.sleep(0.1)
 
            current_pose = self.get_pose()
            dphi = self.normalize_angle(phi - current_pose[2])
            # print(dphi, self.v_left, self.v_right)
        return True
 
 
    def goto(self, coord, phi=None):
        """
        open-loop navigation towards a selected goal, with an optional final heading
      
        Parameters
        ----------
        coord: (float, float)
            coordinates of the robot goal 
        phi: float, optional
            optional final heading of the robot
      
        Returns
        -------
        bool
            True if the destination has been reached, False if the robot has collided
        """
        #TODO: Task01 - Implement the open-loop locomotion control function 
 
        ret = True
        #starting the locomotion thread
        try:
            self.stop = False
            thread1 = thread.Thread(target=self.locomotion)
        except:
            print("Error: unable to start locomotion thread")
            sys.exit(1)
 
        thread1.start()
 
        if not self.rotate_to_target(coord):
            self.stop = True
            thread1.join()
            return False
 
        # move towards target
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        mag_d = float(np.linalg.norm(d))
        direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
        dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
        while mag_d > DISTANCE_THLD:
            if self.robot.get_robot_collision():
                self.stop = True
                thread1.join()
                return False
 
            if np.abs(dphi) > DEGREE_THLD:
                if not self.rotate_to_target(coord):
                    self.stop = True
                    thread1.join()
                    return False
 
            self.steering_lock.acquire()
            self.v_left = -dphi*C_TURNING_SPEED + BASE_SPEED
            self.v_right = dphi*C_TURNING_SPEED + BASE_SPEED
            self.steering_lock.release()
            time.sleep(0.1)
 
            current_pose = self.get_pose()
            d = coord - current_pose[:2]
            mag_d = float(np.linalg.norm(d))
            direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
            dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
  
        if phi is not None:
            return self.rotate_to_direction(phi)
         
        #stop the locomotion thread
        self.stop = True
        thread1.join()
         
        return True
 
    def get_closest_left_and_right(self, direction_vector):
        current_pose = self.get_pose()
        dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
        goal_y = direction_vector["mag"]*np.sin(dphi)
        scan_x, scan_y = self.robot.get_laser_scan()
        ################## CSV DATA COLLECTING
        with open("measures.txt", "a") as myfile:
            myfile.write(", ".join([str(elem) for elem in current_pose]))
            myfile.write("\r\n")
            myfile.write(", ".join([str(elem) for elem in scan_x]))
            myfile.write("\r\n")
            myfile.write(", ".join([str(elem) for elem in scan_y]))
            myfile.write("\r\n")
        ##################
        d = [(np.linalg.norm(np.array([scan_x[i], scan_y[i]])), self.normalize_angle(np.arctan2(scan_y[i], scan_x[i])), i) for i in range(len(scan_x))]
        if len(d) == 0:
            return [10e6, 0, None], [10e6, 0, None]
 
        left = [[v, theta, i] for v, theta, i in d if theta > 0 and not (dphi > 0 and goal_y < scan_y[i] and scan_x[i] > SAFE_DISTANCE)]
        right = [[v, theta, i] for v, theta, i in d if theta <= 0 and not (dphi < 0 and abs(goal_y) < abs(scan_y[i]) and scan_x[i] > SAFE_DISTANCE)]
        # for v, theta, i in d:
            # print(v, theta, goal_y, scan_y[i])
 
        scan_left, scan_right = [10e6, 0, None], [10e6, 0, None]
        if len(left) > 0: scan_left = min(left, key = lambda t: t[0])
        if len(right) > 0: scan_right = min(right, key = lambda t: t[0])
 
        # check which one is closer
        # then check if they are actually the same object (infinitesimal difference of distance between them)
        # if they are the same object, assign it only to the side to which it is closer
        # and look again for the closest object to the other side, on the condition that it is different from the object in question
        if scan_left[2] is not None and scan_right[2] is not None and np.linalg.norm(np.array([scan_x[scan_left[2]] - scan_x[scan_right[2]], scan_y[scan_left[2]] - scan_y[scan_right[2]]])) < 0.1:
            if scan_left[0] < scan_right[0]: scan_right = min(right, key = lambda t: t[0] and np.linalg.norm(np.array([scan_x[scan_left[2]] - scan_x[t[2]], scan_y[scan_left[2]] - scan_y[t[2]]])) > 0.1)
            else: scan_left = min(left, key = lambda t: t[0] and np.linalg.norm(np.array([scan_x[t[2]] - scan_x[scan_right[2]], scan_y[t[2]] - scan_y[scan_right[2]]])) > 0.1)
 
        return scan_left, scan_right
 
    def goto_reactive(self, coord):
        """
        Navigate the robot towards the target with reactive obstacle avoidance 
      
        Parameters
        ----------
        coord: (float, float)
            coordinates of the robot goal 
 
        Returns
        -------
        bool
            True if the destination has been reached, False if the robot has collided
        """
        #TODO: Task02 - Implement the reactive obstacle avoidance
        #starting the locomotion thread
        # if not hasattr(self, "thread1"):
        #     try:
        #         self.stop = False
        #         self.thread1 = thread.Thread(target=self.locomotion)
        #     except:
        #         print("Error: unable to start locomotion thread")
        #         sys.exit(1)
 
        #     self.thread1.start()
         
        try:
            self.stop = False
            thread1 = thread.Thread(target=self.locomotion)
        except:
            print("Error: unable to start locomotion thread")
            sys.exit(1)
 
        thread1.start()
 
        if not self.rotate_to_target(coord):
            self.stop = True
            thread1.join()
            return False
 
        # move towards target
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        direction_vector = {"mag": float(np.linalg.norm(d)), "theta": np.arctan2(d[1], d[0])}
        dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
        scan_left, scan_right = self.get_closest_left_and_right(direction_vector)
        obstacle_last_seen = None
        while direction_vector["mag"] > DISTANCE_THLD:
            if self.robot.get_robot_collision():
                self.stop = True
                thread1.join()
                return False
 
            if np.abs(dphi) > DEGREE_THLD and scan_left[0] > SAFE_DISTANCE and scan_right[0] > SAFE_DISTANCE:
                if obstacle_last_seen is None:
                    obstacle_last_seen = current_pose
                # elif np.abs(self.normalize_angle(obstacle_last_seen[2] - direction_vector["theta"])) > math.pi/2 or np.linalg.norm(obstacle_last_seen[:2] - current_pose[:2]) > 5*SAFE_DISTANCE:
                elif np.linalg.norm(obstacle_last_seen[:2] - current_pose[:2]) > 5*SAFE_DISTANCE:
                    obstacle_last_seen = None
                    if not self.rotate_to_target(coord):
                        self.stop = True
                        thread1.join()
                        return False
 
            # print("pos", current_pose, scan_left[0], scan_right[0])
            if min(scan_left[0], scan_right[0]) < TURN_THLD and direction_vector["mag"] > SAFE_DISTANCE:
                if scan_left[0] < scan_right[0] and scan_left[1] < math.pi/16:
                    self.rotate_to_direction(self.normalize_angle(scan_left[1] + current_pose[2] + math.pi/1.8))
                elif scan_right[1] > -math.pi/16:
                    self.rotate_to_direction(self.normalize_angle(scan_right[1] + current_pose[2] - math.pi/1.8))
            # rotate away from obstacle
 
            self.steering_lock.acquire()
            self.v_left = (1/scan_left[0])*C_AVOID_SPEED + dphi*C_TURNING_SPEED + BASE_SPEED
            self.v_right = (1/scan_right[0])*C_AVOID_SPEED + dphi*C_TURNING_SPEED + BASE_SPEED
            self.steering_lock.release()
            time.sleep(0.1)
 
            current_pose = self.get_pose()
            d = coord - current_pose[:2]
            direction_vector = {"mag": float(np.linalg.norm(d)), "theta": np.arctan2(d[1], d[0])}
            dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
            scan_left, scan_right = self.get_closest_left_and_right(direction_vector)
  
        #stop the locomotion thread
        self.stop = True
        # self.thread1.join()
        thread1.join()
         
        return True       
 
 
if __name__=="__main__":
    robot = Robot()
    #drive the robot into the default position
    robot.turn_on()
    time.sleep(3)
    robot.goto((1,1), phi=math.pi)