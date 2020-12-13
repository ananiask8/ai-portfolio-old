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
            

    def opposite_angle(self, theta):
        return theta + math.pi if theta < 0 else theta - math.pi

    def normalize_angle(self, theta):
        if np.abs(theta) < math.pi:
            return theta

        return theta + 2*math.pi if theta < 0 else theta - 2*math.pi

    def get_far_point_on_goal_vector_line(self, coord, theta, dmin=SPARE_DIST_MIN):
        if theta < math.pi/2:
            phi = theta
            dx = -dmin*np.cos(phi)
            dy = -dmin*np.sin(phi)
        elif theta < math.pi:
            phi = math.pi - theta
            dx = dmin*np.cos(phi)
            dy = -dmin*np.sin(phi)
        elif theta < 3*math.pi/2:
            phi = theta - math.pi
            dx = dmin*np.cos(phi)
            dy = dmin*np.sin(phi)
        else:
            phi = 2*math.pi - theta
            dx = -dmin*np.cos(phi)
            dy = dmin*np.sin(phi)
        return np.array(coord) + np.array([dx, dy])

    def goto_helper(self, coord, phi=None):
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        mag_d = np.abs(np.linalg.norm(d))
        direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
        dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
        # while mag_d > DISTANCE_THLD:
        while mag_d > DISTANCE_THLD or (phi is not None and np.abs(current_pose[2] - phi) > DEGREE_THLD):
            if self.robot.get_robot_collision():
                return False
            print("GOAL", coord, current_pose[:2], mag_d)
            self.steering_lock.acquire()
            self.v_left = -dphi*C_TURNING_SPEED + BASE_SPEED
            self.v_right = dphi*C_TURNING_SPEED + BASE_SPEED
            self.steering_lock.release()
            time.sleep(0.5)

            current_pose = self.get_pose()
            while None in current_pose:
                time.sleep(0.5)
                current_pose = self.get_pose()

            d = coord - current_pose[:2]
            mag_d = np.linalg.norm(d)
            direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
            dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])

        return True

    # def goto(self, coord, phi=None):
    #     """
    #     open-loop navigation towards a selected goal, with an optional final heading
     
    #     Parameters
    #     ----------
    #     coord: (float, float)
    #         coordinates of the robot goal 
    #     phi: float, optional
    #         optional final heading of the robot
     
    #     Returns
    #     -------
    #     bool
    #         True if the destination has been reached, False if the robot has collided
    #     """
    #     #starting the locomotion thread
    #     try:
    #         self.stop = False
    #         thread1 = thread.Thread(target=self.locomotion)
    #     except:
    #         print("Error: unable to start locomotion thread")
    #         sys.exit(1)

    #     thread1.start()

    #     current_pose = self.get_pose()
    #     d = np.array(coord) - current_pose[:2]
    #     mag_d = np.abs(np.linalg.norm(d))

    #     #arctan2 gives correct quadrant
    #     direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}

    #     # if angle is obtuse
    #     if phi is not None and np.abs(direction_vector["theta"] - phi) > math.pi/2:
    #         if not self.goto_helper(self.get_far_point_on_goal_vector_line(coord, phi)):
    #             return False
    #     if not self.goto_helper(coord, phi):
    #         return False
    #     self.stop = True

    #     return True
    #     
    def rotate_to_target(self, coord):
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        theta = np.arctan2(d[1], d[0])
        dphi = self.normalize_angle(theta - current_pose[2])
        while np.abs(dphi) > DEGREE_THLD:
            if self.robot.get_robot_collision():
                self.stop = True
                return False

            self.steering_lock.acquire()
            if dphi > 0:
                self.v_left = -1
                self.v_right = 1
            else:
                self.v_left = 1
                self.v_right = -1
            self.steering_lock.release()
            time.sleep(0.8)

            current_pose = self.get_pose()
            d = coord - current_pose[:2]
            theta = np.arctan2(d[1], d[0])
            dphi = self.normalize_angle(theta - current_pose[2])
            # print(dphi, self.v_left, self.v_right)
        return True

    def rotate_to_final_direction(self, phi):
        current_pose = self.get_pose()
        dphi = self.normalize_angle(phi - current_pose[2])
        while np.abs(dphi) > DEGREE_THLD:
            if self.robot.get_robot_collision():
                self.stop = True
                return False

            self.steering_lock.acquire()
            if dphi > 0:
                self.v_left = -1
                self.v_right = 1
            else:
                self.v_left = 1
                self.v_right = -1
            self.steering_lock.release()
            time.sleep(0.8)

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
        #starting the locomotion thread
        if not hasattr(self, "thread1"):
            try:
                self.stop = False
                self.thread1 = thread.Thread(target=self.locomotion)
            except:
                print("Error: unable to start locomotion thread")
                sys.exit(1)

            self.thread1.start()
        
 
        self.rotate_to_target(coord)

        # move towards target
        current_pose = self.get_pose()
        d = coord - current_pose[:2]
        mag_d = float(np.linalg.norm(d))
        direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
        dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
        while mag_d > DISTANCE_THLD:
            if self.robot.get_robot_collision():
                self.stop = True
                return False

            # Section where I was trying to create a relation between the angle of the vector towards the goal
            # and the angle of the vector that is required to finish
            # I attempted a linear combination of them, so that priority would be given to the appropriate angle
            # The result was that the robot would achieve a state of equilibrium, cycling around the goal, unable to enter it with the desired angle
            # '''
            # alpha = mag_d/(np.abs(current_pose[2] - phi))
            # k = alpha/(W_GOAL + W_DIR*alpha**2) 
            # if np.abs(self.normalize_angle(direction_vector["theta"] - phi)) > math.pi/4.0:
            # # if np.abs(self.normalize_angle(direction_vector["theta"] - phi)) > math.pi/4.0 and np.abs(self.normalize_angle(direction_vector["theta"] - phi)) < 3*math.pi/4.0:
            #     # curve_phi = np.arctan(np.tan((dphi + phi + math.pi)/2.0))
            #     # curve_phi = self.normalize_angle(dphi + phi + math.pi)/2.0
            #     # curve_phi = self.normalize_angle((direction_vector["theta"] + phi + math.pi)/2.0 - current_pose[2])
            #     curve_phi = self.normalize_angle(k*(W_DIR*alpha*direction_vector["theta"] + W_GOAL*(phi+math.pi)/float(alpha)) - current_pose[2])
            # else:
            #     # curve_phi = np.arctan(np.tan((dphi + phi)/2.0))
            #     # curve_phi = self.normalize_angle(dphi + phi)/2.0
            #     # curve_phi = self.normalize_angle((direction_vector["theta"] + phi)/2.0 - current_pose[2])
            #     curve_phi = self.normalize_angle(k*(W_DIR*alpha*direction_vector["theta"] + W_GOAL*phi/float(alpha)) - current_pose[2])
            # '''
            if np.abs(dphi) > DEGREE_THLD:
                self.rotate_to_target(coord)

            self.steering_lock.acquire()
            self.v_left = -dphi*C_TURNING_SPEED + BASE_SPEED
            self.v_right = dphi*C_TURNING_SPEED + BASE_SPEED
            self.steering_lock.release()
            # print(mag_d, current_pose, coord)
            time.sleep(0.8)

            current_pose = self.get_pose()
            d = coord - current_pose[:2]
            mag_d = float(np.linalg.norm(d))
            direction_vector = {"mag": mag_d, "theta": np.arctan2(d[1], d[0])}
            dphi = self.normalize_angle(direction_vector["theta"] - current_pose[2])
 
        if phi is not None:
            return self.rotate_to_final_direction(phi)
 
        # print("DONE", coord, current_pose)

        return True

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
        pass


if __name__=="__main__":
    robot = Robot()
    #drive the robot into the default position
    robot.turn_on()
    time.sleep(3)
    robot.goto((1,1), phi=math.pi)
