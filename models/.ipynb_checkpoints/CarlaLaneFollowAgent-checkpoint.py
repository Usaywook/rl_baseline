from carla2gym.carla.PythonAPI.agents.navigation.controller import VehiclePIDController
from carla2gym.core.utils.transform import *
import numpy as np
import random
import networkx as nx

import yaml
f = open('configs/carla_lane_follow.yaml')
config = yaml.safe_load(f)

class LaneFollowAgent():
    def __init__(self, env):
        self.target_speed = config['target_speed']  # km/h
        self.env = env
        self.mycar = env.actor_info['actors'][0]

        self.vehicle_controller = self.SetController()
        self.lane_follow_way_dist = config['lane_follow_way_dist']

        self.roi1, self.roi2, self.box = self.SetRoI()

    def SetController(self):
        dt_lateral = config['dt_lateral']
        dt_longitudinal = config['dt_longitudinal']
        args_lateral_dict = {
            'K_P': config['lateral_K_P'],
            'K_D': config['lateral_K_D'],
            'K_I': config['lateral_K_I'],
            'dt': dt_lateral}
        args_longitudinal_dict = {
            'K_P': config['longitudinal_K_P'],
            'K_D': config['longitudinal_K_D'],
            'K_I': config['longitudinal_K_I'],
            'dt': dt_longitudinal}
        return VehiclePIDController(self.mycar,
                             args_lateral=args_lateral_dict,
                             args_longitudinal=args_longitudinal_dict)

    def SetRoI(self):
        roi1 = config['roi1']
        roi_box1 = [[roi1[0], roi1[2]], [roi1[0], roi1[3]], [roi1[1], roi1[3]], [roi1[1], roi1[2]]]
        roi2 = config['roi2']
        roi_box2 = [[roi2[0], roi2[2]], [roi2[0], roi2[3]], [roi2[1], roi2[3]], [roi2[1], roi2[2]]]
        box = []
        for x in roi_box1:
            box.append(carla.Location(x[0], x[1], 0.1))
        for x in roi_box2:
            box.append(carla.Location(x[0], x[1], 0.1))
        return roi1, roi2, box

    def showRoI(self):
        for item in self.box:
            self.env.world.debug.draw_string(item + self.mycar.get_location(), 'O', draw_shadow=False,
                                             color=carla.Color(r=255, g=0, b=0), life_time=0.02,
                                             persistent_lines=True)

    def showWay(self, waypoint):
        # BFS to find trajectory
        G = nx.Graph()
        visit = list()
        queue = list()
        queue.append(waypoint)
        visit.append(waypoint)

        while queue:
            u = queue.pop()
            next_waypoints = list(u.next(3.0))
            if not next_waypoints:
                break
            else:
                for v in next_waypoints:
                    if v not in visit:
                        G.add_edge(u, v)
                        visit.append(v)
                        queue.append(v)
            if len(visit) > 20:
                break
        for u,v in list(G.edges()):
            self.env.world.debug.draw_arrow(begin=u.transform.location, end=v.transform.location,
                                            color=carla.Color(r=0,g=0,b=255), life_time=0.001, thickness=0.5)

    def step(self, obs):
        mycar_road_id = obs[3]
        mycar_lane_id = obs[4]
        intersection_dist = obs[5]
        traffic_light = obs[6]

        # is there an obstacle in front of us?
        hazard_detected = False
        # retrieve relevant elements for safe navigation, i.e.: traffic lights
        # and other vehiclesas
        current_waypoint = self.env.map.get_waypoint(self.mycar.get_location(), project_to_road=True,
                                                     lane_type=(carla.LaneType.Driving | carla.LaneType.Shoulder))
        self.showWay(current_waypoint)
        vehicle_state, vehicle = is_vehicle_hazard(self.mycar, self.env.map, self.env.actor_info['actors'])


        # check possible obstacles
        if vehicle_state:
            hazard_detected = True
        # check for the state of the traffic lights
        if traffic_light == 1 or traffic_light == 2:  # red or yellow
            hazard_detected = True

        # lane follow
        next_waypoint = lane_follow(current_waypoint, waypoint_distance=self.lane_follow_way_dist)


        # show roi region
        self.showRoI()

        # apply control
        control = self.vehicle_controller.run_step(self.target_speed, next_waypoint)

        # apply emergency stop
        if hazard_detected:
            control = emergency_stop()

        return np.array([control.steer, control.throttle, control.brake])