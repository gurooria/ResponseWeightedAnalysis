import numpy as np
import robot_env
import random
import utils
import math
import rotations

class FetchEnv(robot_env.RobotEnv):
    """Superclass for all Fetch environments."""
    
    def __init__(self, model_path, image_dims, seed, target_offset, obj_range, target_range, initial_qpos):
        # Initialisations
        self.gripper_extra_height = 0.3
        self.target_in_the_air = True
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.initial_pos_tray = None # of the tray
        self.n_substeps = 10
        self.n_actions = 3 ## because we mask the gripper !!
        self.seed = seed
        self.image_dims = image_dims
        # the init function
        super(FetchEnv, self).__init__(model_path=model_path, n_actions= self.n_actions, initial_qpos=initial_qpos, n_substeps = self.n_substeps)
        np.random.seed(self.seed)
    
    def _set_action(self, action):
        # set the action range for the gripper
        assert action.shape == (3,) # assert means that the condition should be true, otherwise it will raise an error
        action = action.copy()  
        pos_ctrl = action[:3]
        gripper_ctrl = 1 # open fingers
        pos_ctrl *= 0.017  # limit maximum change in position
        rot_ctrl = [1., 0., 1., 0.]  # fixed rotation of the end effector, expressed as a quaternion
        gripper_ctrl = np.array([gripper_ctrl, gripper_ctrl])
        assert gripper_ctrl.shape == (2,)
        action = np.concatenate([pos_ctrl, rot_ctrl, gripper_ctrl])

        # Apply action to simulation
        utils.ctrl_set_action(self.sim, action)
        utils.mocap_set_action(self.sim, action)
        
    def reset(self):
        # Reset the simulation and returns the environment observation in rgb
        did_reset_sim = False
        while not did_reset_sim:
            did_reset_sim = self._reset_sim()

        obs = self._get_obs() # get the observation
        obs_rgb = self.my_render(self.image_dims) # get the rgb observation
        return obs_rgb #, obs
    
    def step(self, action):
        # Take a step in the simulation
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action) # set the action for the gripper
        self.move_object_in_circle()
        
        # get observations
        obs = self._get_obs()
        obs_rgb = self.my_render(self.image_dims)
        done = self._get_terminal() # check if the subpolicy is done
        reward, successful = self.reward()
        if successful: done = True # if the subpolicy is successful, then the episode is done
        
        for _ in range(5): # 5 steps for each action
            self.sim.step()

        return obs_rgb, obs
        
    def _get_terminal(self): # create terminal for each subtask
        # Check if the subpolicy is done
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_pos = self.sim.data.get_site_xpos('object0')
        goal_pos = self.sim.data.get_site_xpos('goal')
        tray_pos = self.sim.data.get_site_xpos('tray')
        timestep = self.sim.data.time * 10
        
        # Check if gripper is too far away from object
        if np.linalg.norm(grip_pos - object_pos) > 0.4:
            return True
        # Check if object is outside the tray
        elif np.linalg.norm(tray_pos - object_pos) > 0.1:
            return True
        return False

            
    def reward(self):
        grip_initial = self.initial_gripper_xpos
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        object_pos = self.sim.data.get_site_xpos('object0')
        goal_pos = self.sim.data.get_site_xpos('goal') #the red tray
        
        dx = grip_pos[0] - object_pos[0]
        dy = grip_pos[1] - object_pos[1]
        dz = grip_pos[2] - object_pos[2]
        
        threshold = 0.015
        # calculate the distance between gripper and object in 3D space
        distance = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2) 

        if distance <= threshold: return 10, True ## indicate if it's succesfull
        return - distance , False

    def _get_obs(self):
        # Get the observation of the environment
        # positions of the gripper, object, goal, and the distance between the object and the goal
        grip_initial = self.initial_gripper_xpos
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
        
        # object positions and velocities
        object_pos = self.sim.data.get_site_xpos('object0')
        # rotations
        object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
        # velocities
        object_velp = self.sim.data.get_site_xvelp('object0') * dt # positional velocity
        object_velr = self.sim.data.get_site_xvelr('object0') * dt # rotational velocity
        # gripper state
        object_rel_pos = object_pos - grip_pos
        object_velp -= grip_velp
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric
        
        # goal position
        goal_pos = self.sim.data.get_body_xpos("goal")
        goal_velp = self.sim.data.get_body_xvelp("goal") * dt # positional velocity of the goal
        goal_velr = self.sim.data.get_body_xvelr("goal") * dt # rotational velocity
        goal_rot = rotations.mat2euler(self.sim.data.get_site_xmat('goal'))
        
        # tray position
        tray_pos = self.sim.data.get_body_xpos("tray")
        tray_velp = self.sim.data.get_body_xvelp("tray") * dt # positional velocity of the tray
        tray_velr = self.sim.data.get_body_xvelr("tray") * dt # rotational velocity
        
        obs = np.concatenate([tray_pos.ravel(), goal_pos.ravel(), tray_velp.ravel(), goal_velp.ravel()])
        
        return obs.copy()
            
    def _viewer_setup(self):
        body_id = self.sim.model.body_name2id('robot0:gripper_link')
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 2.5
        self.viewer.cam.azimuth = 132.
        self.viewer.cam.elevation = -14.

    def _render_callback(self):
        # Visualize target.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
        site_id = self.sim.model.site_name2id('target0')
        self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
        self.sim.forward()
        
    def my_render_video(self): # where we have the channels. array of channels
        # can be used to train the channel, this function should be called in trial.ipynb
        view1 = self.sim.render(width=500, height=500, camera_name='front')
        view1 = np.rot90(view1, 2)
        view2 = self.sim.render(width=500, height=500, camera_name='left')
        view2 = np.rot90(view2, 2)
        return view1[:,:,::-1], view2[:,:,::-1]
    
    def my_render(self, dims): 
        # renders rgb camera views
        view1 = self.sim.render(width=dims, height=dims, camera_name='front')
        view1 = np.rot90(view1, 2)
        view2 = self.sim.render(width=dims, height=dims, camera_name='left')
        view2 = np.rot90(view2, 2)
        return view1, view2      
    
    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.center_of_circle = self.initial_gripper_xpos[:2].copy()

        # we need to put the object inside of the initial tray
        tray_xpos = np.zeros((3, ))
        tray_xpos[0] = self.center_of_circle[0] + np.random.uniform( -0.07, 0.07) # up down
        tray_xpos[1] = self.center_of_circle[1] - 0.35 + 0.3 + np.random.uniform( -0.07, 0.07) # left right .... as you see the front
        tray_xpos[2] = 0.4
        
        # ############ place goal tray
        goal_xpos = np.zeros((3, ))######
        goal_xpos[0] = self.center_of_circle[0] + np.random.uniform( -0.07, 0.07) 
        goal_xpos[1] = self.center_of_circle[1] - 0.35 - 0.05 + np.random.uniform( -0.07, 0.07) ####### the position is different in xml so no need to change
        goal_xpos[2] = 0.4
        
        self.initial_pos_tray = tray_xpos.copy()
        self.initial_pos_goal = goal_xpos.copy()
        
        # set the position of the tray
        x_joint_i = self.sim.model.get_joint_qpos_addr("tray:slidex")
        y_joint_i = self.sim.model.get_joint_qpos_addr("tray:slidey")
         # set the position of the goal
        goal_x_joint_i = self.sim.model.get_joint_qpos_addr("goal:slidex")
        goal_y_joint_i = self.sim.model.get_joint_qpos_addr("goal:slidey")
        
        sim_state = self.sim.get_state()
        
        sim_state.qpos[x_joint_i] = tray_xpos[0]
        sim_state.qpos[y_joint_i] = tray_xpos[1]
        sim_state.qpos[goal_x_joint_i] = goal_xpos[0]
        sim_state.qpos[goal_y_joint_i] = goal_xpos[1]
        self.sim.set_state(sim_state)
        
        # set the start position
        object_xpos = np.zeros((3, ))
        object_xpos[:2] = tray_xpos[:2]
        object_xpos[2] = tray_xpos[2] + 0.02
        object_qpos = self.sim.data.get_joint_qpos('object0:joint')
        assert object_qpos.shape == (7,)
        object_qpos[:3] = object_xpos 
        #object_qpos[0] += random.choice(range(-1,0))
        self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        self.sim.forward()
        
        return True


    def _sample_goal(self, use_fixed_goal=True): 
        """
        use a fixed goal for the target position - may need to be modified
        """
        goal = self.initial_gripper_xpos[:3] + np.random.uniform(-self.target_range, self.target_range, size=3)
        goal += self.target_offset
        goal[2] = self.height_offset
        if self.target_in_the_air and np.random.uniform() < 0.5:
            goal[2] += np.random.uniform(0, 0.45)
        
        if use_fixed_goal:
            goal = self.initial_gripper_xpos[:3] + self.target_offset
            goal[1] += 0.1
            goal[2] = self.height_offset
            goal[2] += 0.3
        return goal.copy()
        

    def _env_setup(self, initial_qpos): 
        for name, value in initial_qpos.items():
            self.sim.data.set_joint_qpos(name, value)
        utils.reset_mocap_welds(self.sim)
        self.sim.forward()
        # Move end effector into position.
        gripper_target = np.array([-0.498, 0.005, -0.431 + self.gripper_extra_height]) + self.sim.data.get_site_xpos('robot0:grip')
        gripper_rotation = np.array([1., 0., 1., 0.])
        self.sim.data.set_mocap_pos('robot0:mocap', gripper_target)
        self.sim.data.set_mocap_quat('robot0:mocap', gripper_rotation)
        for _ in range(10):
            self.sim.step()

        # Extract information for sampling goals.
        self.initial_gripper_xpos = self.sim.data.get_site_xpos('robot0:grip').copy()
        self.height_offset = self.sim.data.get_site_xpos('object0')[2]

    def render(self, mode='human', width=500, height=500):
        return super(FetchEnv, self).render(mode, width, height)
        
    def update_quaternions_by_lissajous(self, timestep, size, alpha, delta, omega, beta):
        t = timestep / 100
        theta_x = omega * t
        theta_y = omega * delta * t + beta
        
        x = size * math.sin(alpha * theta_x)
        y = size * math.sin(alpha * theta_y)

        # Create a rotation matrix to rotate around x and y axis by x and y angles respectively
        rot_matrix_x = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        rot_matrix_y = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])

        # Multiply the rotation matrices to get the final rotation matrix
        rot_matrix = rot_matrix_y.dot(rot_matrix_x)

        # Convert the rotation matrix to a quaternion
        qw = 0.5 * math.sqrt(rot_matrix[0][0] + rot_matrix[1][1] + rot_matrix[2][2] + 1)
        qx = (rot_matrix[2][1] - rot_matrix[1][2]) / (4 * qw)
        qy = (rot_matrix[0][2] - rot_matrix[2][0]) / (4 * qw)
        qz = (rot_matrix[1][0] - rot_matrix[0][1]) / (4 * qw)

        # Normalize the quaternion
        magnitude = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw /= magnitude
        qx /= magnitude
        qy /= magnitude
        qz /= magnitude

        # Return the orientation quaternion
        return [qw, qx, qy, qz]
    
    def move_and_orient(self):
    
        alpha = 1
        beta = 1
        omega = 40 ### this indicates the speed of the curves, bigger means shorter timesteps for the curve!
        delta = 1
        size = 1

        # Get the current timestep
        timestep = self.sim.data.time * 100
        self.sim.step()

        # Update the orientation and position of the tray based on the current timestep
        orientation = self.update_quaternions_by_lissajous(timestep, size, alpha, delta, omega, beta)
        q  = self.sim.data.get_joint_qpos('tray:ball')

        self.sim.data.set_joint_qpos('tray:ball', [q[0], q[1] , q[2], q[1] + 0.1])

        self.sim.data.set_joint_qvel('tray:slidex', 0.3)
        
        
    def move_object_in_circle(self):
        # Tray (green with red ball inside)
        t = self.sim.data.time / 2
        tray = self.sim.data.get_body_xpos("tray")
        x = self.initial_pos_tray[0] - 0.02 + 0.08 * np.sin(t)  # Change the sign of the sine function
        y = self.initial_pos_tray[1] - 0.09 + 0.08 * np.cos(t)
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr('tray:slidex')] = x
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr('tray:slidey')] = y

        # Goal (blue)
        goal = self.sim.data.get_body_xpos("goal")
        x = self.initial_pos_goal[0] - 0.02 + 0.08 * np.sin(t) * 2
        y = self.initial_pos_goal[1] - 0.09 + 0.08 * np.cos(t) * 2
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr('goal:slidex')] = x 
        self.sim.data.qpos[self.sim.model.get_joint_qpos_addr('goal:slidey')] = y
        