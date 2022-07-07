#!/usr/bin/python3

import numpy as np
import math
from qpsolvers import solve_qp
import matplotlib.pyplot as plt

from pickle import NONE
from utils import AgileCommandMode, AgileCommand
from tf.transformations import quaternion_matrix, quaternion_from_matrix
from numpy.linalg import inv
import copy
from tf.transformations import euler_from_quaternion

from gazebo_msgs.msg import ModelStates
from collections import OrderedDict
from geometry_msgs.msg import Pose, Twist, PointStamped
import rospy
from nav_msgs.msg import Odometry# OdometryStamped
from geometry_msgs.msg import TwistStamped

#from rl_example import rl_example
past_pos = np.zeros((10,3))
t_past = 0
dict_old = OrderedDict()

IM_HEIGHT = 320
IM_WIDTH = 240

def compute_command_vision_based(state, img):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command vision-based!")
    # print(state)
    # print("Image shape: ", img.shape)

    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    ################################################
    # !!! End of user code !!!
    ################################################

    return command


def compute_command_state_based(state, obstacles, rl_policy=None):
    ################################################
    # !!! Begin of user code !!!
    # TODO: populate the command message
    ################################################
    print("Computing command based on obstacle information!")
    # print(state)
    # print("Obstacles: ", obstacles)

    # Example of SRT command
    # command_mode = 0
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]
 
    # Example of CTBR command
    # command_mode = 1
    # command = AgileCommand(command_mode)
    # command.t = state.t
    # command.collective_thrust = 10.0
    # command.bodyrates = [0.0, 0.0, 0.0]

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0

    # If you want to test your RL policy
    # if rl_policy is not None:
    #     command = rl_example(state, obstacles, rl_policy)

    ################################################
    # !!! End of user code !!!
    ################################################

    return command

pub = rospy.Publisher('/tuni/obstacle_states', ModelStates, queue_size=10)

def compute_command_cbf_based( state, obstacles , cv_image):


    ###############################################
    global past_pos, t_past, dict_old

    num = obstacles.num
    obs_array = (np.asarray(obstacles.obstacles[:][:]))
    past_pos = np.zeros((num,3))
    
    #print(past_pos.shape)
    #velocity = obstacles.obstacles[:][0].position.y - past_pos
    #past_pos = obs_array
    dt = obstacles.t - t_past
    ob_dict = OrderedDict()
    num_new = 0

    for i in range(num):
        if obs_array[i].position.x > 0:
            num_new +=1
            ob_dict[obs_array[i].scale] = np.array([obs_array[i].position.x, obs_array[i].position.y, obs_array[i].position.z, 0.0, 0.0, 0.0]) #position and liner velcites of osbtacle
    obs_items = list(ob_dict.items())
    #print(items[0])
    ids = []
    poses = []
    vels = []
    for i in range(num_new):
        dist_ = obs_array[i].position.x**2 + obs_array[i].position.y**2 + obs_array[i].position.z**2
        #h.append(dist_ - obs_array[i].scale**2)
        ob_pose = Pose()
        ob_vel = Twist()
        id_ = obs_items[i][0]




        if t_past != 0 and (id_ in dict_old):
            ob_dict[id_][3:6] = np.array([ob_dict[id_][0] -  dict_old[id_][0], ob_dict[id_][1] -  dict_old[id_][1], ob_dict[id_][2] -  dict_old[id_][2]])/dt
            #print(ob_dict[id_])

        ob_pose.position.x = ob_dict[id_][0]
        ob_pose.position.y = ob_dict[id_][1]
        ob_pose.position.z = ob_dict[id_][2]
        ob_vel.linear.x = ob_dict[id_][3]
        ob_vel.linear.y = ob_dict[id_][4]
        ob_vel.linear.z = ob_dict[id_][5]

        ids.append(str(id_))
        poses.append(ob_pose)
        vels.append(ob_vel)

    t_past = obstacles.t
    dict_old = ob_dict
    if len(poses) > 0:
        poses[0].orientation.x = t_past
    obs_msg = ModelStates()
    obs_msg.name = ids
    obs_msg.pose = poses
    obs_msg.twist = vels
    pub.publish(obs_msg)

    ##################################################
    ## TODO just obstacles above be assumed
    ## TODO add multi obstacle to controller function
    ## TODO tunning gains

    #print(cv_image.shape)
    

    
    num = obstacles.num
    obs_array = (np.asarray(obstacles.obstacles[:][:]))
    #print(obs_array[0])

    ##convert world to pixel coordinate ## discusion about approach to have this in ML
    # obs_array[0].position.x
    center_coordinate, r_ = world_to_pixel( drone_state = state, obs_state = obstacles)
    #print(r_)

    ## we are only considering the first obstacle
    circles = [[center_coordinate[0][0], center_coordinate[0][1], r_[0]]]


    result, edges_x, edges_y = CBF_dCBF_creation(circles)

    #print( "pose of first obstacle " + str(center_coordinate[0][0]) + " " + str(center_coordinate[0][1]) + " " + str(r_[0]))

    #print(cv_image[160,120])
    
    output = controller( state , [result[int(cv_image.shape[0]/2),int(cv_image.shape[1]/2)], edges_x[int(cv_image.shape[0]/2),int(cv_image.shape[1]/2)], edges_y[int(cv_image.shape[0]/2),int(cv_image.shape[1]/2)], cv_image[int(cv_image.shape[0]/2),int(cv_image.shape[1]/2)]])
    print("depth value of center of image " + str(cv_image[int(cv_image.shape[0]/2),int(cv_image.shape[1]/2)]))

    #print("min of depth " + str(np.amin(cv_image)))
    #print("max of depth " + str(np.amax(cv_image)))
    print("output of controller ")
    print(output)
    
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.velocity = output
    command.yawrate = 0.0
    #print(command.velocity)
    
    #print("successfull call")

    return command
    


def controller(State, Vision):

    """
    I assumed CBF_2D is based on camera coordinate.
    Z direction is from midpoint of image to forward.
    I also assumed that the states are based on Global coordinate.
    The camera coordinate should be converted to the global coordinate. 

    Image Coordinate:
    1. Origin: center of image on the camera
    2. X: from Camera to forward
    3. Y: Left
    4. Z: Up
    """

    # Robot State in Global Coordinate
    X = State.pos[0]
    Y = State.pos[1]
    Z = State.pos[2]
    

    #### converting state angle from quaternion to euler angles
    qat = State.att
    drone_orientation = [qat[1], qat[2], qat[3], qat[0]]
    roll, pitch, yaw = euler_from_quaternion(drone_orientation)
    thethaa = pitch
    psii = yaw
    phii = roll

    Rrot = np.array([ 
        [math.cos(thethaa)* math.cos(psii)   ,   (-math.cos(phii) * math.sin(psii)) + (math.sin(phii) * math.sin(thethaa) * math.cos(psii))  ,   (math.sin(phii) * math.sin(psii)) + (math.cos(phii) * math.sin(thethaa)* math.cos(psii))],
        [math.cos(thethaa)*math.sin(psii)   ,   (math.cos(phii)* math.cos(psii)) + (math.sin(phii) * math.sin(thethaa) * math.sin(psii))   ,   (-math.sin(phii) * math.cos(psii)) + (math.cos(phii) * math.sin(thethaa) * math.sin(psii))],
        [-math.sin(thethaa) ,   math.sin(phii) * math.cos(thethaa),     math.cos(phii) * math.cos(thethaa)]
     ])

    # Phi = State[3]
    # Theta = State[4]
    # Psi = State[5]


    CBF_2D = Vision[0]
    dCBF_2D = [Vision[1],Vision[2]]
    depth = Vision[3]

    ## hyperparameters ################
    relaxation = 1.0
    C_alpha = 1.0
    P_alpha = 1.0
    C_betha = 150.0
    P_betha = 2.0
    C_gamma = 1.0
    P_gamma = 1.0
    C_eta = 100.0
    P_eta = 1.0
    dXmax = +1.0
    dXmin = -1.0
    dYmax = +2.0
    dYmin = -2.0
    dZmax = +2.0
    dZmin = -2.0
    Delta_ub = +10.0
    Delta_lb = -10.0
    height = 5.0
    ###################################
    
    CBF_2D = CBF_2D / 10000
    CBF = CBF_2D + C_betha*(depth)**P_betha

    print("2d CBF value : " + str(CBF_2D))
    print("CBF value : " + str(CBF))
  
    
    # Image Coordinate to Global Coordinate
    #Rrot = np.identity(3)
    dCBF_2D[0] = dCBF_2D[0] / 100
    dCBF_2D[1] = dCBF_2D[1] / 100
    dCBF = Rrot.dot(np.array([-C_eta*(depth)**P_eta, dCBF_2D[0], dCBF_2D[1]]))
    print("gradient vector value : ")
    print(dCBF)
    ################################### Define Optimization Parameters

    # control = [dX, dY, dZ, delta]
    P = np.diag([1.0, 1.0, 1.0, 10.0])
    q = np.array([-dXmax, 0.0, 0.0, 0.0])
    G = np.array([[1.0, 0, 0, 0],[-1.0, 0, 0, 0],[0, 1.0, 0, 0],[0, -1.0, 0, 0],[0, 0, 1.0, 0],[0, 0, -1.0, 0],[0, 0, 0, 1.0],[0, 0, 0, -1.0],[0, 0, 0, 0],[0, 0, 0, 0]])
    h = np.array([dXmax,-dXmin,dYmax,-dYmin,dZmax,-dZmin,Delta_ub,-Delta_lb,0,0])

    ###################################
    V = 0.5*(0)**2+0.5*(Y-0)**2+0.5*(Z-height)**2

    G[8][0] = -dCBF[0]
    G[8][1] = -dCBF[1]
    G[8][2] = -dCBF[2]
    G[8][3] = 0
    h[8] = C_alpha*CBF**P_alpha

    G[9][0] = 0
    G[9][1] = Y
    G[9][2] = (Z-height)
    G[9][3] = -1
    h[9] = -C_gamma*V**P_gamma
    
    
    # output is in Global Coordinate
    [dX, dY, dZ, Delta] = solve_qp(P, q, G, h)

    output = [dX, dY, dZ]
    return output


def CBF_dCBF_creation( circles):

    gridd = np.zeros((IM_HEIGHT, IM_WIDTH, 2))
    #immi = np.zeros((320,240))
    for i in range( 0, IM_HEIGHT):
        for j in range( 0, IM_WIDTH):
            gridd[ i, j, 0] = i
            gridd[ i, j, 1] = j
    immi = np.zeros((IM_HEIGHT, IM_WIDTH, len(circles)))
    for i, [center_x, center_y, radius] in enumerate(circles):

        immi[:,:,i] = (np.sum((gridd - np.array([center_y, center_x]))**2, axis = -1)) - radius**2

    result = np.min(immi, axis = 2)

    edges_x, edges_y = np.gradient(result.T)

    return result, edges_x, edges_y



def project_3d_pixel(c2o_point):
    K = np.asarray([[228.504*0.75, 0.1, 160, 0],[0.0, 228.504*0.75, 120, 0],[0.0, 0.0, 1.0, 0.0]])
    z = K @ c2o_point
    z = z/z[2]
    #print(z)
    return z



def world_to_pixel( drone_state, obs_state):

    
    ############################ defining camera parameters #############################
    w_to_d = np.eye(4)
    d_to_w = np.eye(4)
    w_t_d_translation = np.ones((4,1))

    camera_matrix = np.eye(3)
    camera_matrix[0,0] = 228.504 #4.57008
    camera_matrix[1,1] = 228.504 #* 1.3333
    camera_matrix[0,2] = 160
    camera_matrix[1,2] = 120

    t1 = np.asarray([[0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.3],
                    [0.0, 0.0, 0.0, 1.0]
                    ])
    t2 = np.asarray([[1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, -1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0]
                    ])

    d_to_c = t1 @ t2

    c_to_d = inv(d_to_c)

    l = []
    center_coordinates = []
    r_ = []

    ####################################################################################

    obstacle_states = copy.deepcopy(obs_state)

    #obs_array = (np.asarray(obs_state.obstacles[:][:]))
    #print(obs_state.obstacles[:][1])

    qat = drone_state.att
    pose = drone_state.pos
    w_t_d_translation = np.array([pose[0] , pose[1], pose[2], 1]).T
    rot_mat = quaternion_matrix([qat[0], qat[1], qat[2], qat[3]])
    w_to_d = rot_mat
    w_to_d[ :, -1] = w_t_d_translation
    
    d_to_w = inv(w_to_d)
    
    #print("obstacle")
    for i in range(obstacle_states.num):
        # obstacle_states.pose[i].position.y = 
        w_to_o = np.eye(4)
        w_to_o[0, -1] = obs_state.obstacles[:][i].position.x + drone_state.pos[0]
        w_to_o[1, -1] =obs_state.obstacles[:][i].position.y + drone_state.pos[1]
        w_to_o[2, -1] = obs_state.obstacles[:][i].position.z + drone_state.pos[2]

        d_to_o = d_to_w @ w_to_o
        obstacle_states.obstacles[:][i].position.x = d_to_o[0, -1]
        obstacle_states.obstacles[:][i].position.y = d_to_o[1, -1]
        obstacle_states.obstacles[:][i].position.z = d_to_o[2, -1]

        

    num = (obstacle_states.num)   ## this will be equal to 10 obstacles each time
    
    ## info of obstacles and time
    info = np.zeros((num, 3))
    count_in = 0
    count_out = 0   

    for i in range(num):
        d_to_o = np.eye(4)
        d_to_o[0, -1] = obstacle_states.obstacles[:][i].position.x
        d_to_o[1, -1] = obstacle_states.obstacles[:][i].position.y
        d_to_o[2, -1] = obstacle_states.obstacles[:][i].position.z
        # test_points
        # d_to_o[0, -1] = test_points[i][0]
        # d_to_o[1, -1] = test_points[i][1]
        # d_to_o[2, -1] = test_points[i][2]

        c_to_o = c_to_d @ d_to_o
        
            
    ## checking if obstacle is in images view, so we check that if positions when converted to pixel coordinate is between the valid range
        
        
        scale = float(obstacle_states.obstacles[:][i].scale)
        th_dist = 5.1 + scale + scale*0.8
        #print("th_dist: ",th_dist)
        if (obstacle_states.obstacles[:][i].position.x < th_dist): 

            #l,_ = cv2.projectPoints(np.asarray(test_points[i], dtype=np.float),rvec, tvec, camera_matrix, np.asarray([]))
            pixel_coordinate = project_3d_pixel(c_to_o[:,-1])
            l = np.asarray([[0,0]])
            l[0,0] = pixel_coordinate[0] 
            l[0,1] = pixel_coordinate[1]

            center_coordinates.append((int(l[0,0]), int(l[0,1])))

            r = (float(obstacle_states.obstacles[:][i].scale)/obstacle_states.obstacles[:][i].position.x)*180
            r_.append(int(r))
        #print(center_coordinates)
        return center_coordinates, r_
    
