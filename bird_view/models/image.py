import math

import numpy as np

import torch
import torch.nn as nn

from . import common
from .agent import Agent
from .controller import CustomController, PIDController
from .controller import ls_circle
import matplotlib.pyplot as plt
import carla
from enum import Enum
from shapely.geometry import Polygon
from collections import deque

from agents.navigation.local_planner import LocalPlanner
from agents.navigation.global_route_planner import GlobalRoutePlanner
from agents.tools.misc import draw_waypoints, get_speed, is_within_distance, get_trafficlight_trigger_location, compute_distance

from bird_view.models.birdview import BirdViewPolicyModelSS


CROP_SIZE = 192
STEPS = 5
COMMANDS = 4
DT = 0.1
CROP_SIZE = 192
PIXELS_PER_METER = 5
PIXEL_OFFSET = 10

        
class ImagePolicyModelSS(common.ImageNetResnetBase):
    def __init__(self, backbone, warp=False, pretrained=False, all_branch=False, **kwargs):
        super().__init__(backbone, pretrained=pretrained, input_channel=3, bias_first=False)
        
        self.c = {
                'resnet18': 512,
                'resnet34': 512,
                'resnet50': 2048
                }[backbone]
        self.warp = warp
        self.rgb_transform = common.NormalizeV2(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        self.deconv = nn.Sequential(
            nn.BatchNorm2d(2*self.c + 2*128),
            nn.ConvTranspose2d(2*self.c + 2*128,256,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256,128,3,2,1,1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128,64,3,2,1,1),
            nn.ReLU(True),
        )
        
        if warp:
            ow,oh = 48,48
        else:
            ow,oh = 96,40 
        
        self.location_pred = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm2d(64),
                nn.Conv2d(64,STEPS,1,1,0),
                common.SpatialSoftmax(ow,oh,STEPS),
            ) for i in range(4)
        ])
        
        self.all_branch = all_branch
        

    def forward(self, image_left, image_right, velocity, command):
        if self.warp:
            warped_image_left = tgm.warp_perspective(image_left, self.M, dsize=(192, 192))
            resized_image_left = resize_images(image_left)
            image_left = torch.cat([warped_image_left, resized_image_left], 1)

            warped_image_right = tgm.warp_perspective(image_right, self.M, dsize=(192, 192))
            resized_image_right = resize_images(image_right)
            image_right = torch.cat([warped_image_right, resized_image_right], 1)
        
        
        image_left = self.rgb_transform(image_left)
        image_right = self.rgb_transform(image_right)

        h_l= self.conv_left(image_left)
        h_r= self.conv_right(image_right)
        # h = torch.cat((h_l, h_r), dim=1)
        b, c, kh, kw = h_l.size()
        
        # Late fusion for velocity
        velocity = velocity[...,None,None,None].repeat((1,128,kh,kw))
        
        h = torch.cat((h_l, velocity, h_r, velocity), dim=1)
        h = self.deconv(h)

        location_preds = [location_pred(h) for location_pred in self.location_pred]
        location_preds = torch.stack(location_preds, dim=1)
        location_pred = common.select_branch(location_preds, command)

        if self.all_branch:
            return location_pred, location_preds
        
        return location_pred


class ImageAgent(Agent):
    def __init__(self, vehicle, steer_points=None, pid=None, gap=5, camera_args={'x':384,'h':160,'fov':90,'world_y':1.4,'fixed_offset':4.0}, **kwargs):
        super().__init__(**kwargs)
        self._vehicle = vehicle
        self._world = self._vehicle.get_world()
        self._map = self._world.get_map()
        self.fixed_offset = float(camera_args['fixed_offset'])
        print ("Offset: ", self.fixed_offset)
        w = float(camera_args['x'])
        h = float(camera_args['h'])
        self.img_size = np.array([w,h])
        self.gap = gap
        self.model_pred = None
        self.model_preds = None
        
        if steer_points is None:
            steer_points = {"1": 4, "2": 3, "3": 2, "4": 2}

        if pid is None:
            pid = {
                "1" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Left
                "2" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Right
                "3" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Straight
                "4" : {"Kp": 1.0, "Ki": 0.10, "Kd":0.0}, # Follow
            }

        # if pid is None:
        #     pid = {
        #         "1" : {"Kp": 1.0, "Ki": 0.0, "Kd":0.0}, # Left
        #         "2" : {"Kp": 1.0, "Ki": 0.0, "Kd":0.0}, # Right
        #         "3" : {"Kp": 1.0, "Ki": 0.0, "Kd":0.0}, # Straight
        #         "4" : {"Kp": 1.0, "Ki": 0.0, "Kd":0.0}, # Follow
        #     }

        self.steer_points = steer_points
        self.turn_control = CustomController(pid)
        self.speed_control = PIDController(K_P= 1.0, K_I=0.1, K_D=2.5) # 0.2, 0.0, 0.0
        
        self.engine_brake_threshold = 2.0
        self.brake_threshold = 2.0
        
        self.last_brake = -1

        self._local_planner = LocalPlanner(self._vehicle)
        self._global_planner = GlobalRoutePlanner(self._map, 2.0)

        # self.birdview_net_pred = None
        # self.birdview_net_model = BirdViewPolicyModelSS(backbone='resnet18', all_branch=True)
        # self.birdview_net_model.load_state_dict(torch.load('/media/storage/karthik/lbc/ckpts/priveleged/model-128.th'))
        # self.birdview_net_model = self.birdview_net_model.to(self.device)
        # self.birdview_net_model.eval()
        self.dx = 0
        self.dy = -PIXEL_OFFSET
        self.crop_size = CROP_SIZE
        self.center_x, self.center_y = 160, 260-self.crop_size//2
        self.debug_list = []
        

    def run_step(self, observations, teaching=False):
        self._local_planner.run_step()
        rgb_left = observations['rgb_left'].copy()
        rgb_right = observations['rgb_right'].copy()
        # birdview = observations['birdview'].copy()
        # birdview = birdview[self.dy+self.center_y-self.crop_size//2:self.dy+self.center_y+self.crop_size//2,self.dx+self.center_x-self.crop_size//2:self.dx+self.center_x+self.crop_size//2]

        speed = np.linalg.norm(observations['velocity'])
        _cmd = int(self._local_planner.command.value)
        command = self.one_hot[int(_cmd) - 1]

        with torch.no_grad():
            _rgb_left = self.transform(rgb_left).to(self.device).unsqueeze(0)
            _rgb_right = self.transform(rgb_right).to(self.device).unsqueeze(0)
            # _birdview = self.transform(birdview).to(self.device).unsqueeze(0)
            _speed = torch.FloatTensor([speed]).to(self.device)
            _command = command.to(self.device).unsqueeze(0)
            if self.model.all_branch:
                model_pred, model_preds = self.model(_rgb_left, _rgb_right, _speed, _command)
            else:
                model_pred = self.model(_rgb_left, _rgb_right, _speed, _command)
            # birdview_net_pred, birdview_net_preds = self.birdview_net_model(_birdview, _speed, _command)


        model_pred = model_pred.squeeze().detach().cpu().numpy()
        model_preds = model_preds.squeeze().detach().cpu().numpy()
        # birdview_net_pred = birdview_net_pred.squeeze().detach().cpu().numpy()

        self.model_pred = model_pred
        self.model_preds = model_preds
        # self.birdview_net_pred = birdview_net_pred

        pixel_pred = model_pred

        # Project back to world coordinate
        model_pred = (model_pred+1)*self.img_size/2

        world_pred = self.unproject(model_pred)

        targets = [(0, 0)]

        for i in range(STEPS):
            pixel_dx, pixel_dy = world_pred[i]
            angle = np.arctan2(pixel_dx, pixel_dy)
            dist = np.linalg.norm([pixel_dx, pixel_dy])
            # print(angle)

            targets.append([dist * np.cos(angle), dist * np.sin(angle)])

        targets = np.array(targets)

        # if np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() > 2.5:
        #     target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / ((self.gap * DT))
        # else:
        #     target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / ((self.gap * DT))
        c, r = ls_circle(targets)
        n = self.steer_points.get(str(_cmd), 1)
        closest = common.project_point_to_circle(targets[n], c, r)
        v = [1.0, 0.0, 0.0]
        w = [closest[0], closest[1], 0.0]
        alpha = common.signed_angle(v, w)
        # print('alpha', alpha)
        target_speed = np.linalg.norm(targets[:-1] - targets[1:], axis=1).mean() / ((self.gap * DT))

        acceleration = target_speed - speed
        self.debug_list.append([targets, _cmd, acceleration, target_speed, speed])

        

        steer = self.turn_control.run_step(alpha, _cmd)
        if target_speed > 50/3.6:
            acceleration = 0.0
        throttle = self.speed_control.step(acceleration)
        brake = 0.0

        # Slow or stop.
        if target_speed <= self.engine_brake_threshold:
            steer = 0.0
            throttle = 0.0
        
        if target_speed <= self.brake_threshold:
            brake = 1.0
            
        self.debug = {
                # 'curve': curve,
                'target_speed': target_speed,
                'target': closest,
                'locations_world': targets,
                'locations_pixel': model_pred.astype(int),
                }
        
        control = self.postprocess(steer, throttle, brake)
        if teaching:
            return control, pixel_pred
        else:
            return control

    def unproject(self, output, world_y=1.4, fov=90):

        cx, cy = self.img_size / 2
        
        w, h = self.img_size
        
        f = w /(2 * np.tan(fov * np.pi / 360))
        
        xt = (output[...,0:1] - cx) / f
        yt = (output[...,1:2] - cy) / f
        
        world_z = world_y / yt
        world_x = world_z * xt
        
        world_output = np.stack([world_x, world_z],axis=-1)
        
        if self.fixed_offset:
            world_output[...,1] -= self.fixed_offset
        
        world_output = world_output.squeeze()
        
        return world_output

    def set_destination(self, end_location, start_location=None):
        """
        This method creates a list of waypoints between a starting and ending location,
        based on the route returned by the global router, and adds it to the local planner.
        If no starting location is passed, the vehicle local planner's target location is chosen,
        which corresponds (by default), to a location about 5 meters in front of the vehicle.

            :param end_location (carla.Location): final location of the route
            :param start_location (carla.Location): starting location of the route
        """
        if not start_location:
            start_location = self._local_planner.target_waypoint.transform.location
            clean_queue = True
        else:
            start_location = self._vehicle.get_location()
            clean_queue = False

        start_waypoint = self._map.get_waypoint(start_location)
        end_waypoint = self._map.get_waypoint(end_location)

        route_trace = self.trace_route(start_waypoint, end_waypoint)
        self._local_planner.set_global_plan(route_trace, clean_queue=clean_queue)

    def set_global_plan(self, plan, stop_waypoint_creation=True, clean_queue=True):
        """
        Adds a specific plan to the agent.

            :param plan: list of [carla.Waypoint, RoadOption] representing the route to be followed
            :param stop_waypoint_creation: stops the automatic random creation of waypoints
            :param clean_queue: resets the current agent's plan
        """
        self._local_planner.set_global_plan(
            plan,
            stop_waypoint_creation=stop_waypoint_creation,
            clean_queue=clean_queue
        )

    def trace_route(self, start_waypoint, end_waypoint):
        """
        Calculates the shortest route between a starting and ending waypoint.

            :param start_waypoint (carla.Waypoint): initial waypoint
            :param end_waypoint (carla.Waypoint): final waypoint
        """
        start_location = start_waypoint.transform.location
        end_location = end_waypoint.transform.location
        return self._global_planner.trace_route(start_location, end_location)

    def done(self):
        """Check whether the agent has reached its destination."""
        np.save('a.npy', np.array(self.debug_list,dtype=object), allow_pickle=True)
        return self._local_planner.done()
