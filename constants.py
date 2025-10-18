import numpy as np

# Keyframes for the "laying" posture.
# The ordering below is chosen to match the actuator order defined in your XML:
#   - Free root: position (0-2) and orientation (3-6)
#   - Trunk joints: abdomen_y, abdomen_z, abdomen_x (7-9)
#   - Right leg: right_hip_x, right_hip_z, right_hip_y, right_knee (10-13)
#   - Left leg: left_hip_x, left_hip_z, left_hip_y, left_knee (14-17)
#   - Right arm: right_shoulder1, right_shoulder2, right_elbow (18-20)
#   - Left arm: left_shoulder1, left_shoulder2, left_elbow (21-23)
KEYFRAMES = {
    "supine": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.1,
        # Root orientation (quaternion)
        0, 0.7071, 0., 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        0,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        0,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0,
        # right_shoulder2
        -np.pi/4,
        # right_elbow
        -np.pi/2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        np.pi/4,
        # left_elbow
        -np.pi/2
    ]),
"prone": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.14,
        # Root orientation (quaternion)
        0, -0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        0,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        0,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        -np.pi / 4,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        np.pi / 4,
        # left_elbow
        -np.pi / 2
    ]),
    "prone_2": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.08,
        # Root orientation (quaternion)
        0, -0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        0,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        0,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        -np.pi/4,
        # right_shoulder2
        -np.pi/3, #-np.pi / 4,
        # right_elbow
        -np.pi / 4,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        np.pi/4,
        # left_shoulder2
        np.pi/3, #np.pi / 4,
        # left_elbow
        -np.pi / 4
    ]),
    #npi/2 npi/6 npi/4 
    #npi/4 npi/3 npi/4 
    #0 pi/4 pi/2 

    #let shoulder1 rotate from 0 to 2pi 
    #just make sure we can get the other two to work. 
    #np.pi/2 and np.pi/12 and npi/2 works 
    #np.pi/4 and np.pi/4 and np/2 works 
    #npi/4 npi/3 and npi/4 works arms out to side elbows bent 
    "seated": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.55,
        # Root orientation (quaternion)
        0, -0.20, 0., 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        -0,
        # abdomen_x: rotation about the x-axis
        0.,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        -np.pi / 4,
        # right_hip_y
        - 2 * np.pi / 3,
        # right_knee
        0.,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        -np.pi / 4,
        # left_hip_y
        - 2 * np.pi / 3,
        # left_knee
        0.,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        np.pi / 4,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        -np.pi / 4,
        # left_elbow
        -np.pi / 2
    ]),
    "knees": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 1.0,
        # Root orientation (quaternion)
        0.7071, 0, 0., 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        0.,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        0,
        # left_elbow
        -np.pi / 2
    ]),
     #root pos .9 absz - .3 knees - np.pi/2 - .3 
    #root pos .8, abz -.6 and knees -np.pi/2 - .6 
    #rot pos .7 abz - .9 knees - np.pi/2 - .9 
    #rot pos .6 abz - 1.2 knees - np.pi/2 - 1.2 
    #root pos 1 absz 0  knees np.pi/2
    "knees_left": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 1.0,
        # Root orientation (quaternion)
        0, 0, 0, 1, #facing left 
        #1, 0, 0, 0, #facing right 
        #0.7071, 0, 0., -0.7071, #facing 6 o clock 
        #0.7071, 0, 0., 0.7071, #facing 12 o clokc 
        
        #0, 0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        0.,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        0,
        # left_elbow
        -np.pi / 2
    ]),
    "standing_left": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 1.3,
        # Root orientation (quaternion)
        0, 0, 0, 1, #facing left 
        #1, 0, 0, 0, #facing right 
        #0.7071, 0, 0., -0.7071, #facing 6 o clock 
        #0.7071, 0, 0., 0.7071, #facing 12 o clokc 
        
        #0, 0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0.,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        0.,
        # right_knee
        0.,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        0,
        # left_knee
        0.,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        0.,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        0,
        # left_elbow
        -np.pi / 2
    ]),
    "crawl": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.6,
        # Root orientation (quaternion)
        0, -0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        -np.pi / 4,
        # abdomen_x: rotation about the x-axis
        0.,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        -np.pi / 6,
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        -np.pi / 6,
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        -np.pi / 4, #-np.pi/2, #
        # right_shoulder2
        np.pi / 4,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        np.pi / 4, #np.pi/2, #
        # left_shoulder2
        -np.pi / 4,
        # left_elbow
        -np.pi / 2
    ]),
    "crawl_perpendicular": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.7,
        # Root orientation (quaternion)
        0, -0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        -np.pi / 4,
        # abdomen_x: rotation about the x-axis
        0.,

        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        -np.pi / 4, #changed from /6 
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        -np.pi / 4, #changed from /6
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        -np.pi/12, 
        # right_shoulder2
        np.pi / 4,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        np.pi/12,
        # left_shoulder2
        -np.pi / 4,
        # left_elbow
        -np.pi / 2
    ]),
     "crawl_more_perpendicular": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 0.7,
        # Root orientation (quaternion)
        0, -0.7071, 0, 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        -np.pi / 4,
        # abdomen_x: rotation about the x-axis
        0.,

        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        -np.pi / 4, #changed from /6 
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        -np.pi / 4, #changed from /6
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0, 
        # right_shoulder2
        np.pi / 4,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        -np.pi / 4,
        # left_elbow
        -np.pi / 2
    ]),
    "lunge_left": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 1.0,
        # Root orientation (quaternion)
        0.7071, 0, 0., 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        -np.pi / 2,
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        np.pi / 6,
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        0.,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        0,
        # left_elbow
        -np.pi / 2
    ]),
    "lunge_right": np.array([
        # --- Free Root ---
        # Root position (x, y, z)
        0., 0., 1.0,
        # Root orientation (quaternion)
        0.7071, 0, 0., 0.7071,
        
        # --- Trunk (Abdomen) Joints ---
        # abdomen_y: rotation about the y-axis
        0.,
        # abdomen_z: rotation about the z-axis
        0,
        # abdomen_x: rotation about the x-axis
        0,
        
        # --- Right Leg Joints ---
        # right_hip_x
        0.,
        # right_hip_z
        0.,
        # right_hip_y
        np.pi / 6,
        # right_knee
        -np.pi / 2,
        
        # --- Left Leg Joints ---
        # left_hip_x
        0,
        # left_hip_z
        0.,
        # left_hip_y
        -np.pi / 2,
        # left_knee
        -np.pi / 2,
        
        # --- Right Arm Joints ---
        # right_shoulder1
        0.,
        # right_shoulder2
        0.,
        # right_elbow
        -np.pi / 2,
        
        # --- Left Arm Joints ---
        # left_shoulder1
        0,
        # left_shoulder2
        0,
        # left_elbow
        -np.pi / 2
    ]),
}
