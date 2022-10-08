#!/usr/bin/env python

## import ros packages
import rospy
from geometry_msgs.msg import Pose

## import fundamental packages
import numpy as np
import math
import random

# epsilon for testing whether a number is close to zer
_EPS = np.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def vector3_to_array(vector_msg):
    output = np.zeros(3)
    output[0] = vector_msg.x
    output[1] = vector_msg.y
    output[2] = vector_msg.z

    return output

def quat_to_array(quat_msg):
    output = np.zeros(4)
    output[0] = quat_msg.x
    output[1] = quat_msg.y
    output[2] = quat_msg.z
    output[3] = quat_msg.w

    return output

def array_to_posmsg(position, orientation):
    msg = Pose()
    msg.position.x = position[0]
    msg.position.y = position[1]
    msg.position.z = position[2]
    msg.orientation.x = orientation[0]
    msg.orientation.y = orientation[1]
    msg.orientation.z = orientation[2]
    msg.orientation.w = orientation[3]

    return msg

def translation_matrix(direction):
    """Return matrix to translate by direction vector.
    >>> v = numpy.random.random(3) - 0.5
    >>> numpy.allclose(v, translation_matrix(v)[:3, 3])
    True
    """
    M = np.identity(4)
    M[:3, 3] = direction[:3]
    return M

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    >>> R = quaternion_matrix([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(R, rotation_matrix(0.123, (1, 0, 0)))
    True
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)

def euler_matrix(ai, aj, ak, axes='sxyz'):
    """Return homogeneous rotation matrix from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> R = euler_matrix(1, 2, 3, 'syxz')
    >>> numpy.allclose(numpy.sum(R[0]), -1.34786452)
    True
    >>> R = euler_matrix(1, 2, 3, (0, 1, 0, 1))
    >>> numpy.allclose(numpy.sum(R[0]), -0.383436184)
    True
    >>> ai, aj, ak = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    >>> for axes in _TUPLE2AXES.keys():
    ...    R = euler_matrix(ai, aj, ak, axes)
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak

    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci*ck, ci*sk
    sc, ss = si*ck, si*sk

    M = np.identity(4)
    if repetition:
        M[i, i] = cj
        M[i, j] = sj*si
        M[i, k] = sj*ci
        M[j, i] = sj*sk
        M[j, j] = -cj*ss+cc
        M[j, k] = -cj*cs-sc
        M[k, i] = -sj*ck
        M[k, j] = cj*sc+cs
        M[k, k] = cj*cc-ss
    else:
        M[i, i] = cj*ck
        M[i, j] = sj*sc-cs
        M[i, k] = sj*cc+ss
        M[j, i] = cj*sk
        M[j, j] = sj*ss+cc
        M[j, k] = sj*cs-sc
        M[k, i] = -sj
        M[k, j] = cj*si
        M[k, k] = cj*ci
    return M

def translation_from_matrix(matrix):
    """Return translation vector from translation matrix.
    >>> v0 = numpy.random.random(3) - 0.5
    >>> v1 = translation_from_matrix(translation_matrix(v0))
    >>> numpy.allclose(v0, v1)
    True
    """
    return np.array(matrix, copy=False)[:3, 3].copy()

def quaternion_from_matrix(matrix):
    """Return quaternion from rotation matrix.
    >>> R = rotation_matrix(0.123, (1, 2, 3))
    >>> q = quaternion_from_matrix(R)
    >>> numpy.allclose(q, [0.0164262, 0.0328524, 0.0492786, 0.9981095])
    True
    """
    q = np.empty((4, ), dtype=np.float64)
    M = np.array(matrix, dtype=np.float64, copy=False)[:4, :4]
    t = np.trace(M)
    if t > M[3, 3]:
        q[3] = t
        q[2] = M[1, 0] - M[0, 1]
        q[1] = M[0, 2] - M[2, 0]
        q[0] = M[2, 1] - M[1, 2]
    else:
        i, j, k = 0, 1, 2
        if M[1, 1] > M[0, 0]:
            i, j, k = 1, 2, 0
        if M[2, 2] > M[i, i]:
            i, j, k = 2, 0, 1
        t = M[i, i] - (M[j, j] + M[k, k]) + M[3, 3]
        q[i] = t
        q[j] = M[i, j] + M[j, i]
        q[k] = M[k, i] + M[i, k]
        q[3] = M[k, j] - M[j, k]
    q *= 0.5 / math.sqrt(t * M[3, 3])
    return q

def quaternion_from_euler(ai, aj, ak, axes='sxyz'):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> numpy.allclose(q, [0.310622, -0.718287, 0.444435, 0.435953])
    True
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    if frame:
        ai, ak = ak, ai
    if parity:
        aj = -aj

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = math.cos(ai)
    si = math.sin(ai)
    cj = math.cos(aj)
    sj = math.sin(aj)
    ck = math.cos(ak)
    sk = math.sin(ak)
    cc = ci*ck
    cs = ci*sk
    sc = si*ck
    ss = si*sk

    quaternion = np.empty((4, ), dtype=np.float64)
    if repetition:
        quaternion[i] = cj*(cs + sc)
        quaternion[j] = sj*(cc + ss)
        quaternion[k] = sj*(cs - sc)
        quaternion[3] = cj*(cc - ss)
    else:
        quaternion[i] = cj*sc - sj*cs
        quaternion[j] = cj*ss + sj*cc
        quaternion[k] = cj*cs - sj*sc
        quaternion[3] = cj*cc + sj*ss
    if parity:
        quaternion[j] *= -1

    return quaternion

def quaternion_multiply(quaternion1, quaternion0):
    """Return multiplication of two quaternions.
    >>> q = quaternion_multiply([1, -2, 3, 4], [-5, 6, 7, 8])
    >>> numpy.allclose(q, [-44, -14, 48, 28])
    True
    """
    x0, y0, z0, w0 = quaternion0
    x1, y1, z1, w1 = quaternion1
    return np.array((
         x1*w0 + y1*z0 - z1*y0 + w1*x0,
        -x1*z0 + y1*w0 + z1*x0 + w1*y0,
         x1*y0 - y1*x0 + z1*w0 + w1*z0,
        -x1*x0 - y1*y0 - z1*z0 + w1*w0), dtype=np.float64)

def euler_from_matrix(matrix, axes='sxyz'):
    """Return Euler angles from rotation matrix for specified axis sequence.
    axes : One of 24 axis sequences as string or encoded tuple
    Note that many Euler angle triplets can describe one matrix.
    >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    >>> R1 = euler_matrix(al, be, ga, 'syxz')
    >>> numpy.allclose(R0, R1)
    True
    >>> angles = (4.0*math.pi) * (numpy.random.random(3) - 0.5)
    >>> for axes in _AXES2TUPLE.keys():
    ...    R0 = euler_matrix(axes=axes, *angles)
    ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    ...    if not numpy.allclose(R0, R1): print axes, "failed"
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        _ = _TUPLE2AXES[axes]
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i+parity]
    k = _NEXT_AXIS[i-parity+1]

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def euler_from_quaternion(quaternion, axes='sxyz'):
    """Return Euler angles from quaternion for specified axis sequence.
    >>> angles = euler_from_quaternion([0.06146124, 0, 0, 0.99810947])
    >>> numpy.allclose(angles, [0.123, 0, 0])
    True
    """
    return euler_from_matrix(quaternion_matrix(quaternion), axes)

def initialize_model_locations(obs_info=None):
    # get parameters
    # get ros parameters
    env_radius = rospy.get_param("/environment/radius", 6.0)
    discomfort_dist = rospy.get_param("/environment/discomfort_dist", 0.25)
    agent_radius = rospy.get_param("/agent/robot/radius", 0.3)
    agent_kinematics = rospy.get_param("/agent/robot/kinematics", "holonomic")
    obs_radius = rospy.get_param("/obstacle/human/radius", 0.3)
    obs_v_pref = rospy.get_param("/obstacle/human/v_pref", 1.0)
    randomize_attributes = rospy.get_param("/obstacle/human/randomize_attributes", False)
    ## total number of obstacles
    num_obs = rospy.get_param("/obstacle/num_obstacle", 5)
    ## number of static obstacles
    num_static = rospy.get_param("/obstacle/num_static", 0)
    ## number of static obstacles is smaller than or equal to the total number of obstacles
    num_static = min(num_static, num_obs)
    if rospy.has_param("/agent/initial_position"):
        agent_init_pos = rospy.get_param("/agent/initial_position")
    else:
        agent_init_pos = []
    if rospy.has_param("/obstacle/initial_position_list"):
        obs_init_pos_list = rospy.get_param("/obstacle/initial_position_list")
    else:
        obs_init_pos_list = []
    if rospy.has_param("/obstacle/object_name_list"):
        object_name_list = rospy.get_param("/obstacle/object_name_list")
    else:
        object_name_list = []
    ## load skin list for human actors
    actor_skin_list = rospy.get_param("/obstacle/visual/actor_skin_list")
    model_path_dict = rospy.get_param("/obstacle/visual/model_path_dict")
    #####################################################################################
    ## INITIALIZE AEGNT ##
    #####################################################################################
    # initialize position of agent
    if len(agent_init_pos) == 0:
        if agent_kinematics == 'unicycle':
            angle = np.random.uniform(0, np.pi * 2)
            px = env_radius * np.cos(angle)
            py = env_radius * np.sin(angle)
            while True:
                gx, gy = np.random.uniform(-env_radius, env_radius, 2)
                if np.linalg.norm([px - gx, py - gy]) >= env_radius:
                    break
        # randomize starting position and goal position
        else:
            while True:
                px, py, gx, gy = np.random.uniform(-env_radius, env_radius, 4)
                if np.linalg.norm([px - gx, py - gy]) >= env_radius:
                    break
    else:
        px, py = agent_init_pos
        while True:
            gx, gy = np.random.uniform(-env_radius, env_radius, 2)
            if np.linalg.norm([px - gx, py - gy]) >= env_radius:
                break
    # finalize agent initial position and goal
    agent_pos = np.array([px, py])
    agent_goal = np.array([gx, gy])
    #####################################################################################

    #####################################################################################
    ## INITIALIZE OBSTACLE ##
    #####################################################################################
    ## number of objects is smaller than or equal to the number of static obstacles
    num_obj = min(len(object_name_list), num_static)
    ## initialize obstacle attributes
    attributes = np.ones([num_obs, 2])*[obs_radius, obs_v_pref]
    # get randomized attributes of human
    if randomize_attributes:
        ## randomize size: default_radius +- 0.1
        attributes[num_obj:,0] += np.random.uniform(size=num_obs - num_obj)*0.2 - 0.1
        attributes[num_obj:,0] = np.maximum(attributes[num_obj:,0], 0.1)
        ## randomize preferred velocity: default_velocity +- 0.5
        attributes[num_obj:,1] += np.random.uniform(size=num_obs - num_obj) - 0.5
        attributes[num_obj:,1] = np.maximum(attributes[num_obj:,1], 0.0)
        
    if obs_info is not None:
        for i in range(num_obs):
            attributes[i,0] = float(obs_info[i]["size"])

    ## set initial positions of obstacles
    num_init = len(obs_init_pos_list)
    init_pos_arr = np.zeros([num_obs, 2])
    if num_init > 0:
        init_pos_arr[:num_init,:] = np.array(obs_init_pos_list)
    for i in range(num_init, num_obs):
        is_collision = True
        while is_collision:
            angle = np.random.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            obs_v_pref = 1.0 if attributes[i,1] == 0 else attributes[i,1]
            noise = (np.random.random(2) - 0.5) * obs_v_pref
            pos = env_radius * np.array([np.cos(angle), np.sin(angle)]) + noise
            
            for j in range(num_init + 2):
                if j < num_init:
                    pos_other = init_pos_arr[j]
                    min_dist = attributes[j,0] + attributes[i,0] + discomfort_dist
                elif j == num_init:
                    pos_other = agent_pos
                    min_dist = agent_radius + attributes[i,0] + discomfort_dist
                else:
                    pos_other = agent_goal
                    min_dist = agent_radius + attributes[i,0] + discomfort_dist
                dist = np.linalg.norm(pos - pos_other)
                # if any other obstacles bump into the current obstacle at current initial position,
                # quit computing distance and find another initial position of the current obstacle
                if dist < min_dist:
                    is_collision = True
                    break
                else:
                    is_collision = False

        # append the newly found initial position
        init_pos_arr[i,:] = pos
        num_init += 1

    goal = np.zeros([num_obs, 2])
    # initialize goals of obstacles
    for i in range(num_obs):
        if i < num_static:
            goal_ = init_pos_arr[i,:]
        else:
            goal_ = -init_pos_arr[i,:]
            # avoid setting static objects' goals as current object's goal
            if num_static > 0:
                is_collision = True
                while is_collision:
                    dist_arr = np.linalg.norm(init_pos_arr[:num_static,:] - goal_, axis=1)
                    idx_violated = dist_arr < attributes[:num_static,0] + attributes[i,0] + discomfort_dist
                    
                    if np.sum(idx_violated) > 0:
                        goal_ += np.random.randn(2)
                    else:
                        is_collision = False
        # store goals
        goal[i,:] = goal_
    #####################################################################################

    #####################################################################################
    ## SET ROSPARAM ##
    #####################################################################################
    # arrange agent information
    agent_info = {}
    agent_info["init_pos"] = agent_pos.tolist()
    agent_info["goal"] = agent_goal.tolist()

    # arrange obstacle information
    obs_info = []
    for i in range(num_obs):
        obs_dict = {}
        obs_dict["name"] = "obstacle_%d"%i
        obs_dict["is_static"] = i < num_static
        # check if the obstacle is human and set its model path
        is_human = False
        if i < num_obj:
            model_path = None
            for j in range(len(model_path_dict)):
                if object_name_list[i] == model_path_dict[j]["name"]:
                    model_path = model_path_dict[j]["path"]
            if model_path is None:
                rospy.loginfo("[generate_obstacle_world.py] model named %s not found. Instead, the obstacle will be set as a human"%object_name_list[i])
                is_human = True
            else:
                obs_dict["model_path"] = model_path
        else:
            is_human = True
            
        obs_dict["is_human"] = is_human
        if is_human:
            obs_dict["model_path"] = random.choice(actor_skin_list)
            
        obs_dict["size"] = float(attributes[i,0])
        obs_dict["v_pref"] = float(attributes[i,1])
        obs_dict["init_pos"] = init_pos_arr[i].tolist()
        obs_dict["goal"] = goal[i].tolist()
        obs_info.append(obs_dict)
    #####################################################################################
    
    return agent_info, obs_info