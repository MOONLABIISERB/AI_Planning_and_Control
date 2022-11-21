#!/usr/bin/env python

# Copyright (c) 2018 Intel Labs.
# authors: German Ros (german.ros@intel.com)
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""Example of automatic vehicle control from client side."""

from __future__ import print_function

import matplotlib.pyplot as plt

import argparse
import collections
import datetime
import glob
import logging
import math
import os
from threading import local
from tkinter import Image
import numpy.random as random
import re
import sys
import weakref
from pathlib import Path
import queue
import tqdm
import lmdb
try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError(
        'cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Find CARLA module ---------------------------------------------------------
# ==============================================================================
try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass
sys.path.append('/media/storage/karthik/lbc')
# ==============================================================================
# -- Add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
    pass

import carla
from carla import ColorConverter as cc

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from agents.navigation.local_planner1 import LocalPlannerNew, RoadOption
import inspect


# ==============================================================================
# -- Global functions ----------------------------------------------------------
# ==============================================================================
try:
    sys.path.append('/opt/carla-simulator')
except IndexError:
    pass
import bird_view.utils.bz_utils as bzu
from bird_view.models.image import ImageAgent
from bird_view.utils.map_utils import Wrapper as map_utils
BACKGROUND = [0, 47, 0]
COLORS = [
        (102, 102, 102),
        (253, 253, 17),
        (204, 6, 5),
        (250, 210, 1),
        (39, 232, 51),
        (0, 0, 142),
        (220, 20, 60)
        ]


def visualize_birdview(birdview):
    """
    0 road
    1 lane
    2 red light
    3 yellow light
    4 green light
    5 vehicle
    6 pedestrian
    """
    h, w = birdview.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[...] = BACKGROUND

    for i in range(len(COLORS)):
        canvas[birdview[:,:,i] > 0] = COLORS[i]

    return canvas

def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []
def get_birdview(observations):
    birdview = [
            observations['road'],
            observations['lane'],
            observations['traffic'],
            observations['vehicle'],
            observations['pedestrian']
            ]
    birdview = [x if x.ndim == 3 else x[...,None] for x in birdview]
    birdview = np.concatenate(birdview, 2)

    return birdview

def process(observations):
    result = dict()
    result['rgb_left'] = observations['rgb_left'].copy()
    result['rgb_right'] = observations['rgb_right'].copy()
    result['birdview'] = observations['birdview'].copy()
    control = observations['control']
    control = [control.steer, control.throttle, control.brake]

    result['control'] = np.float32(control)
    measurements = [
        observations['position'],
        observations['orientation'],
        observations['velocity'],
        observations['acceleration'],
        observations['command'].value,
        observations['control'].steer,
        observations['control'].throttle,
        observations['control'].brake,
        observations['control'].manual_gear_shift,
        observations['control'].gear
        ]
    measurements = [x if isinstance(x, np.ndarray) else np.float32([x]) for x in measurements]
    measurements = np.concatenate(measurements, 0)

    result['measurements'] = measurements

    return result

def spawn_traffic(client, args):
    vehicles_list = []
    walkers_list = []
    all_id = []
    synchronous_master = False
    world = client.get_world()
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_global_distance_to_leading_vehicle(2.5)
    settings = world.get_settings()

    traffic_manager.set_synchronous_mode(True)
    if not settings.synchronous_mode:
        synchronous_master = True
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
    else:
        synchronous_master = False
    
    world.apply_settings(settings)
    # -------------
    # Spawn Vehicles
    # -------------
    blueprints = get_actor_blueprints(world, 'vehicle.*', 'All')
    blueprintsWalkers = get_actor_blueprints(world, 'walker.pedestrian.*','2')

    blueprints = sorted(blueprints, key=lambda bp: bp.id)

    spawn_points = world.get_map().get_spawn_points()
    number_of_spawn_points = len(spawn_points)

    if args.n_vehicles < number_of_spawn_points:
        random.shuffle(spawn_points)
    elif args.n_vehicles > number_of_spawn_points:
        msg = 'requested %d vehicles, but could only find %d spawn points'
        logging.warning(msg, args.n_vehicles, number_of_spawn_points)
        args.n_vehicles = number_of_spawn_points

    SpawnActor = carla.command.SpawnActor
    SetAutopilot = carla.command.SetAutopilot
    FutureActor = carla.command.FutureActor

    batch = []
    for n, transform in enumerate(spawn_points):
        if n >= args.n_vehicles:
            break
        blueprint = random.choice(blueprints)
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        if blueprint.has_attribute('driver_id'):
            driver_id = random.choice(blueprint.get_attribute('driver_id').recommended_values)
            blueprint.set_attribute('driver_id', driver_id)
        else:
            blueprint.set_attribute('role_name', 'autopilot')
        batch.append(SpawnActor(blueprint, transform)
            .then(SetAutopilot(FutureActor, True, traffic_manager.get_port())))

    for response in client.apply_batch_sync(batch, synchronous_master):
        if response.error:
            logging.error(response.error)
        else:
            vehicles_list.append(response.actor_id)
    # -------------
    # Spawn Walkers
    # -------------
    percentagePedestriansRunning = 0.0      # how many pedestrians will run
    percentagePedestriansCrossing = 60.0     # how many pedestrians will walk through the road
    # 1. take all the random locations to spawn
    spawn_points = []
    for i in range(args.n_pedestrians):
        spawn_point = carla.Transform()
        loc = world.get_random_location_from_navigation()
        if (loc != None):
            spawn_point.location = loc
            spawn_points.append(spawn_point)
    # 2. we spawn the walker object
    batch = []
    walker_speed = []
    for spawn_point in spawn_points:
        walker_bp = random.choice(blueprintsWalkers)
        # set as not invincible
        if walker_bp.has_attribute('is_invincible'):
            walker_bp.set_attribute('is_invincible', 'false')
        # set the max speed
        if walker_bp.has_attribute('speed'):
            if (random.random() > percentagePedestriansRunning):
                # walking
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[1])
            else:
                # running
                walker_speed.append(walker_bp.get_attribute('speed').recommended_values[2])
        else:
            print("Walker has no speed")
            walker_speed.append(0.0)
        batch.append(SpawnActor(walker_bp, spawn_point))
    results = client.apply_batch_sync(batch, True)
    walker_speed2 = []
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list.append({"id": results[i].actor_id})
            walker_speed2.append(walker_speed[i])
    walker_speed = walker_speed2
    # 3. we spawn the walker controller
    batch = []
    walker_controller_bp = world.get_blueprint_library().find('controller.ai.walker')
    for i in range(len(walkers_list)):
        batch.append(SpawnActor(walker_controller_bp, carla.Transform(), walkers_list[i]["id"]))
    results = client.apply_batch_sync(batch, True)
    for i in range(len(results)):
        if results[i].error:
            logging.error(results[i].error)
        else:
            walkers_list[i]["con"] = results[i].actor_id
    # 4. we put together the walkers and controllers id to get the objects from their id
    for i in range(len(walkers_list)):
        all_id.append(walkers_list[i]["con"])
        all_id.append(walkers_list[i]["id"])
    all_actors = world.get_actors(all_id)

    # 5. initialize each controller and set target to walk to (list is [controler, actor, controller, actor ...])
    # set how many pedestrians can cross the road
    world.set_pedestrians_cross_factor(percentagePedestriansCrossing)
    for i in range(0, len(all_id), 2):
        # start walker
        all_actors[i].start()
        # set walk to random point
        all_actors[i].go_to_location(world.get_random_location_from_navigation())
        # max speed
        all_actors[i].set_max_speed(float(walker_speed[int(i/2)]))

    print('spawned %d vehicles and %d walkers, press Ctrl+C to exit.' % (len(vehicles_list), len(walkers_list)))
    # Example of how to use Traffic Manager parameters
    traffic_manager.global_percentage_speed_difference(30.0)

    # while True:
    #     if synchronous_master:
    #         world.tick()
    #     else:
    #         world.wait_for_tick()



    return 

def carla_img_to_np(carla_img):
    carla_img.convert(cc.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:,:,:3]
    img = img[:,:,::-1]

    return img

def is_within_distance_ahead(target_location, current_location, orientation, max_distance, degree=60):
    u = np.array([
        target_location.x - current_location.x,
        target_location.y - current_location.y])
    distance = np.linalg.norm(u)

    if distance > max_distance:
        return False

    v = np.array([
        math.cos(math.radians(orientation)),
        math.sin(math.radians(orientation))])

    angle = math.degrees(math.acos(np.dot(u, v) / distance))

    return angle < degree

def find_weather_presets():
    """Method to find weather presets"""
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    def name(x): return ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    """Method to get actor display name"""
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name

VEHICLE_NAME = 'vehicle.ford.mustang'

PRESET_WEATHERS = {
    1: carla.WeatherParameters.ClearNoon,
    2: carla.WeatherParameters.CloudyNoon,
    3: carla.WeatherParameters.WetNoon,
    4: carla.WeatherParameters.WetCloudyNoon,
    5: carla.WeatherParameters.MidRainyNoon,
    6: carla.WeatherParameters.HardRainNoon,
    7: carla.WeatherParameters.SoftRainNoon,
    8: carla.WeatherParameters.ClearSunset,
    9: carla.WeatherParameters.CloudySunset,
    10: carla.WeatherParameters.WetSunset,
    11: carla.WeatherParameters.WetCloudySunset,
    12: carla.WeatherParameters.MidRainSunset,
    13: carla.WeatherParameters.HardRainSunset,
    14: carla.WeatherParameters.SoftRainSunset,
}

# ==============================================================================
# -- World ---------------------------------------------------------------
# ==============================================================================

class World(object):
    """ Class representing the surrounding environment """

    def __init__(self, carla_world, hud, args, vehicle_name=VEHICLE_NAME, client=None, respwan_peds=True, big_cam=False):
        """Constructor method"""
        self._args = args
        self.world = carla_world
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
            sys.exit(1)
        self.hud = hud
        self.player = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.camera_manager = None
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self._actor_filter = vehicle_name #args.filter

        ## CAMERA PARAMETERS
        self.rgb_queue_left= None
        self.rgb_image_left = None
        self.rgb_queue_right = None
        self.rgb_image_right = None
        self.big_cam = big_cam
        self.big_cam_queue = None

        self.disable_two_wheels = False
        self.client = client
        self.weather_index = args.weather
        # self.command = RoadOption.LANEFOLLOW
        self.rgb_camera_bp_left = None
        self.rgb_camera_bp_right = None
        self._start_pose = None
        # print(self.command)
        self.restart(args)
        self.world.on_tick(hud.on_world_tick)
        self.recording_enabled = False
        self.recording_start = 0

    def restart(self, args):
        """Restart the world"""
        vehicles = self.world.get_actors().filter('vehicle.*')
        for v in vehicles:
            _ = v.destroy()
        walkers = self.world.get_actors().filter('walker.*')
        for w in walkers:
            w.destroy()
        # Keep same camera config if the camera manager exists.
        cam_index = self.camera_manager.index if self.camera_manager is not None else 0
        cam_pos_id = self.camera_manager.transform_index if self.camera_manager is not None else 0

        # Get a random blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter(self._actor_filter))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)

        # Spawn the player.
        if self.player is not None:
            spawn_point = self.player.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.player = self.world.try_spawn_actor(blueprint, spawn_point)
            self.modify_vehicle_physics(self.player)
        while self.player is None:
            if not self.map.get_spawn_points():
                print('There are no spawn points available in your map/town.')
                print('Please add some Vehicle Spawn Point to your UE4 scene.')
                sys.exit(1)
            # spawn_points = self.map.get_spawn_points()
            self._start_pose = random.choice(self.map.get_spawn_points())
            self.player = self.world.try_spawn_actor(blueprint, self._start_pose)
            self.modify_vehicle_physics(self.player)

        # _ = spawn_traffic(self.client, self._args)

        self.rgb_queue_left = queue.Queue()
        if self.big_cam:
            self.big_cam_queue = queue.Queue()
            rgb_camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_camera_bp.set_attribute('image_size_x', '800')
            rgb_camera_bp.set_attribute('image_size_y', '600')
            rgb_camera_bp.set_attribute('fov', '90')
            big_camera = self.world.spawn_actor(
                rgb_camera_bp,
                carla.Transform(carla.Location(x=1.0, z=1.4), carla.Rotation(pitch=0)),
                attach_to=self.player)
            big_camera.listen(self.big_cam_queue.put)
        
        ## NOTE: camera settings for a normal camera. Default is True
        rgb_camera_bp_left = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp_left.set_attribute('image_size_x', '384')
        rgb_camera_bp_left.set_attribute('image_size_y', '160')
        rgb_camera_bp_left.set_attribute('fov', '90')
        self.rgb_camera_bp_left = self.world.spawn_actor(
            rgb_camera_bp_left,
            carla.Transform(carla.Location(x=2.0, z=1.4, y=-0.5), carla.Rotation(pitch=0)),
            attach_to=self.player)

        self.rgb_camera_bp_left.listen(self.rgb_queue_left.put)
        self.rgb_queue_right = queue.Queue()
        ## NOTE: camera settings for a normal camera. Default is True
        rgb_camera_bp_right = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_camera_bp_right.set_attribute('image_size_x', '384')
        rgb_camera_bp_right.set_attribute('image_size_y', '160')
        rgb_camera_bp_right.set_attribute('fov', '90')
        self.rgb_camera_bp_right = self.world.spawn_actor(
            rgb_camera_bp_right,
            carla.Transform(carla.Location(x=2.0, z=1.4, y=0.5), carla.Rotation(pitch=0)),
            attach_to=self.player)

        self.rgb_camera_bp_right.listen(self.rgb_queue_right.put)

        # SET WEATHER
        self.world.set_weather(PRESET_WEATHERS[self.weather_index])

        if self._args.sync:
            self.world.tick()
        else:
            self.world.wait_for_tick()

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.transform_index = cam_pos_id
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)
        map_utils.init(self.client, self.world, self.map, self.player)


    def spawn_pedestrians(self, n_pedestrians):
        SpawnActor = carla.command.SpawnActor

        peds_spawned = 0
        
        self.walkers = []
        self.controllers = []

        walkers = []
        controllers = []
        
        while peds_spawned < n_pedestrians:
            spawn_points = []
            _walkers = []
            _controllers = []
            
            for i in range(n_pedestrians - peds_spawned):
                spawn_point = carla.Transform()
                loc = self.world.get_random_location_from_navigation()
    
                if loc is not None:
                    spawn_point.location = loc
                    spawn_points.append(spawn_point)
    
            blueprints = self.world.get_blueprint_library().filter('walker.pedestrian.*')
            batch = []
            for spawn_point in spawn_points:
                walker_bp = random.choice(blueprints)
    
                if walker_bp.has_attribute('is_invincible'):
                    walker_bp.set_attribute('is_invincible', 'false')
    
                batch.append(SpawnActor(walker_bp, spawn_point))
    
            for result in self.client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    peds_spawned += 1
                    _walkers.append(result.actor_id)
    
            walker_controller_bp = self.world.get_blueprint_library().find('controller.ai.walker')
            batch = [SpawnActor(walker_controller_bp, carla.Transform(), walker) for walker in _walkers]
    
            for result in self.client.apply_batch_sync(batch, True):
                if result.error:
                    print(result.error)
                else:
                    _controllers.append(result.actor_id)
                    
            controllers.extend(_controllers)
            walkers.extend(_walkers)
        
        for i in range(len(controllers)):
            self.controllers = self.world.get_actors(controllers)
            self.walkers = self.world.get_actors(walkers)
        for i in range(0, len(self.controllers)):
            # start walker
            self.controllers[i].start()
            # set walk to random point
            self.controllers[i].go_to_location(self.world.get_random_location_from_navigation())
            # random max speed
            self.controllers[i].set_max_speed(1 + random.random())


        print ("spawned %d pedestrians"%len(controllers))

        return self.world.get_actors(walkers), self.world.get_actors(controllers)

    # def next_weather(self, reverse=False):
    #     """Get next weather setting"""
    #     self._weather_index += -1 if reverse else 1
    #     self._weather_index %= len(self._weather_presets)
    #     preset = self._weather_presets[self._weather_index]
    #     self.hud.notification('Weather: %s' % preset[1])
    #     self.player.get_world().set_weather(preset[0])

    def modify_vehicle_physics(self, actor):
        #If actor is not a vehicle, we cannot use the physics control
        try:
            physics_control = actor.get_physics_control()
            physics_control.use_sweep_wheel_collision = True
            actor.apply_physics_control(physics_control)
        except Exception:
            pass

    def tick(self, clock, agent):
        """Method for every tick"""
        # self.command = self._local_planner.checkpoint[1]
        map_utils.tick()
        while self.rgb_image_left is None or self.rgb_queue_left.qsize() > 0:
            self.rgb_image_left = self.rgb_queue_left.get()
        while self.rgb_image_right is None or self.rgb_queue_right.qsize() > 0:
            self.rgb_image_right = self.rgb_queue_right.get()
        
        if self.big_cam:
            while self.big_cam_image is None or self.big_cam_queue.qsize() > 0:
                self.big_cam_image = self.big_cam_queue.get()
        self.hud.tick(self, clock, agent)
        
    def render(self, display):
        """Render world"""
        # map_utils.render_world()
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy_sensors(self):
        """Destroy sensors"""
        if self.rgb_queue:
            with self.rgb_queue.mutex:
                self.rgb_queue.queue.clear()
        
        if self.big_cam_queue:
            with self.big_cam_queue.mutex:
                self.big_cam_queue.queue.clear()
        self.camera_manager.sensor.destroy()
        self.camera_manager.sensor = None
        self.camera_manager.index = None

    def destroy(self):
        """Destroys all actors"""
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.gnss_sensor.sensor,
            self.player,
            self.rgb_camera_bp_right,
            self.rgb_camera_bp_left]
        for actor in actors:
            if actor is not None:
                actor.destroy()

    ## NOTE: CARLA FUNCTIONS
    def get_observations(self, agent):
        result = dict()
        result.update(map_utils.get_observations())
        pos = self.player.get_location()
        ori = self.player.get_transform().get_forward_vector()
        vel = self.player.get_velocity()
        acc = self.player.get_acceleration()
        command = agent._local_planner.command
        control = self.player.get_control()

        # self.command = local_planner.command
        
        result.update({
                'position': np.float32([pos.x, pos.y, pos.z]),
                'orientation': np.float32([ori.x, ori.y]),
                'velocity': np.float32([vel.x, vel.y, vel.z]),
                'acceleration': np.float32([acc.x, acc.y, acc.z]),
                'command': command if command is not None else RoadOption.LANEFOLLOW,
                'control': control,
                })
        # print ("%.3f, %.3f"%(self.rgb_image.timestamp, self._world.get_snapshot().timestamp.elapsed_seconds))
        result.update({
            'rgb_left': carla_img_to_np(self.rgb_image_left),
            'rgb_right': carla_img_to_np(self.rgb_image_right),
            'birdview': get_birdview(result),
            # 'collided': self.collided
            })

        if self.big_cam:
            result.update({
                'big_cam': carla_img_to_np(self.big_cam_image),
            })
        
        # plt.imshow(np.concatenate((result['rgb_left'], result['rgb_right'])))
        # plt.pause(0.01)
        return result




# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================


class KeyboardControl(object):
    def __init__(self, world):
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True

    @staticmethod
    def _is_quit_shortcut(key):
        """Shortcut for quitting"""
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """Class for HUD text"""

    def __init__(self, width, height):
        """Constructor method"""
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()
        self.map = None

    def on_world_tick(self, timestamp):
        """Gets informations from the world at every tick"""
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock, agent):
        """HUD method for every tick"""
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        transform = world.player.get_transform()
        vel = world.player.get_velocity()
        control = world.player.get_control()
        heading = 'N' if abs(transform.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(transform.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > transform.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > transform.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        pedestrians = world.world.get_actors().filter('walker.*')
        self.map = visualize_birdview(get_birdview(world.get_observations(agent)))
        self.rgb_image_left = world.get_observations(agent)['rgb_left']
        self.rgb_image_right = world.get_observations(agent)['rgb_right']

        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name.split('/')[-1],
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (transform.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (transform.location.x, transform.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % transform.location.z,
            '']
        if isinstance(control, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', control.throttle, 0.0, 1.0),
                ('Steer:', control.steer, -1.0, 1.0),
                ('Brake:', control.brake, 0.0, 1.0),
                ('Reverse:', control.reverse),
                ('Hand brake:', control.hand_brake),
                ('Manual:', control.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(control.gear, control.gear)]
        elif isinstance(control, carla.WalkerControl):
            self._info_text += [
                ('Speed:', control.speed, 0.0, 5.556),
                ('Jump:', control.jump)]
        # self._info_text += ['Command: % 20s' % world.player._local_planner.command.value]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles),
            'Number of pedestrians: % 5d' % len(pedestrians)]


        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']

        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        vehicles = [(dist(x.get_location()), x) for x in vehicles if x.id != world.player.id]


        for dist, vehicle in sorted(vehicles):
            if dist > 200.0:
                break
            vehicle_type = get_actor_display_name(vehicle, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, vehicle_type))
            
        def dist(l):
            return math.sqrt((l.x - transform.location.x)**2 + (l.y - transform.location.y)
                             ** 2 + (l.z - transform.location.z)**2)
        if len(pedestrians) > 1:
            self._info_text += ['Nearby pedestrians:']

        pedestrians = [(dist(x.get_location()), x) for x in pedestrians if x.id != world.player.id]

        for dist, pedestrian in sorted(pedestrians):
            if dist > 200.0:
                break
            pedestrian_type = get_actor_display_name(pedestrian, truncate=22)
            self._info_text.append('% 4dm %s' % (dist, pedestrian_type))

    def toggle_info(self):
        """Toggle info on or off"""
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """Notification text"""
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """Error text"""
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """Render for HUD class"""
        if self._show_info:
            map_surface = pygame.Surface((320, 320))
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            # display.blit(pygame.surfarray.make_surface(self.rgb_image_left), (0, 0))
            # display.blit(pygame.surfarray.make_surface(self.rgb_image_right), (160, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        fig = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + fig * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (fig * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """ Class for fading text """

    def __init__(self, font, dim, pos):
        """Constructor method"""
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """Set fading text"""
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        """Fading text method for every tick"""
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """Render fading text method"""
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """ Helper class for text render"""

    def __init__(self, font, width, height):
        """Constructor method"""
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for i, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, i * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """Toggle on or off the render help"""
        self._render = not self._render

    def render(self, display):
        """Render help text method"""
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================


class CollisionSensor(object):
    """ Class for collision sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        """Gets the history of collisions"""
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        """On collision method"""
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x ** 2 + impulse.y ** 2 + impulse.z ** 2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)

# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================


class LaneInvasionSensor(object):
    """Class for lane invasion sensors"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        """On invasion method"""
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================


class GnssSensor(object):
    """ Class for GNSS sensors"""

    def __init__(self, parent_actor):
        """Constructor method"""
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        blueprint = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(blueprint, carla.Transform(carla.Location(x=1.0, z=2.8)),
                                        attach_to=self._parent)
        # We need to pass the lambda a weak reference to
        # self to avoid circular reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        """GNSS method"""
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude

# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================


class CameraManager(object):
    """ Class for camera management"""

    def __init__(self, parent_actor, hud):
        """Constructor method"""
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        bound_y = 0.5 + self._parent.bounding_box.extent.y
        attachment = carla.AttachmentType
        self._camera_transforms = [
            (carla.Transform(carla.Location(x=2.0, z=1.4), carla.Rotation(pitch=0)),attachment.Rigid),
            (carla.Transform(
                carla.Location(x=1.6, z=1.7)), attachment.Rigid),
            (carla.Transform(
                carla.Location(x=5.5, y=1.5, z=1.5)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-8.0, z=6.0), carla.Rotation(pitch=6.0)), attachment.SpringArm),
            (carla.Transform(
                carla.Location(x=-1, y=-bound_y, z=0.5)), attachment.Rigid)]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            blp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                blp.set_attribute('image_size_x', str(hud.dim[0]))
                blp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                blp.set_attribute('range', '50')
            item.append(blp)
        self.index = None

    def toggle_camera(self):
        """Activate a camera"""
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.set_sensor(self.index, notify=False, force_respawn=True)

    def set_sensor(self, index, notify=True, force_respawn=False):
        """Set a sensor"""
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None else (
            force_respawn or (self.sensors[index][0] != self.sensors[self.index][0]))
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index][0],
                attach_to=self._parent,
                attachment_type=self._camera_transforms[self.transform_index][1])

            # We need to pass the lambda a weak reference to
            # self to avoid circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        """Get the next sensor"""
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        """Toggle recording on or off"""
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        """Render method"""
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 4), 4))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data)  # pylint: disable=assignment-from-no-return
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)

# ==============================================================================
# -- Game Loop ---------------------------------------------------------
# ==============================================================================


def game_loop(args):
    """
    Main loop of the simulation. It handles updating all the HUD information,
    ticking the agent and, if needed, the world.
    """

    pygame.init()
    pygame.font.init()
    world = None
    

    try:
        save_dir = Path(args.dataset_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        for i in tqdm.tqdm(range(args.n_episodes), desc='Episode'):
            
            if args.seed:
                random.seed(args.seed)
            # model_path = Path(args.model_path)
            # config = bzu.load_json(str(model_path.parent / 'config.json'))

            # agent_maker = _agent_factory_hack(model_path, config, False)

            client = carla.Client(args.host, args.port)
            client.set_timeout(4.0)

            # traffic_manager = client.get_trafficmanager()
            sim_world = client.load_world(args.town)

            if args.sync:
                original_settings = sim_world.get_settings()
                settings = sim_world.get_settings()
                if not settings.synchronous_mode:
                    settings.synchronous_mode = True
                    settings.fixed_delta_seconds = 0.05
                sim_world.apply_settings(settings)

                traffic_manager = client.get_trafficmanager()
                traffic_manager.set_synchronous_mode(True)

            display = pygame.display.set_mode(
                (args.width, args.height),
                pygame.HWSURFACE | pygame.DOUBLEBUF)

            
            hud = HUD(args.width, args.height)
            world = World(carla_world=client.get_world(), hud=hud, args=args, client=client)
            controller = KeyboardControl(world)
            _ = spawn_traffic(client=client, args=args)
            if args.agent == "Basic":
                agent = BasicAgent(world.player, target_speed=30)
            elif args.agent == "LBC":
                agent = ImageAgent(world.player, model_path=Path(args.model_path))
            else:
                print("behaviour agent")
                agent = BehaviorAgent(world.player, behavior=args.behavior)
                agent.set_target_speed(20)
                agent.follow_speed_limits(False)

            # Set the agent destination
            spawn_points = world.map.get_spawn_points()
            destination = random.choice(spawn_points).location
            agent.set_destination(destination)
            data = list()
            progress = tqdm.tqdm(range(args.frames_per_episode), desc='Frame')

            

            filepath = save_dir.joinpath('%03d' % i)

            if filepath.exists():
                if world is not None:
                    settings = world.world.get_settings()
                    settings.synchronous_mode = False
                    settings.fixed_delta_seconds = None
                    world.world.apply_settings(settings)
                    traffic_manager.set_synchronous_mode(True)

                    world.destroy()
                continue




            clock = pygame.time.Clock()

            while len(data) < args.frames_per_episode:
                for _ in range(args.frames_skip):

                    clock.tick()
                    if args.sync:
                        world.world.tick()
                    else:
                        world.world.wait_for_tick()
                    if controller.parse_events():
                        return

                    world.tick(clock, agent)
                    world.render(display)
                    pygame.display.flip()

                    if agent.done():
                        if True:
                            agent.set_destination(random.choice(spawn_points).location)
                            world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
                            print("The target has been reached, searching for another target")
                    #     else:
                    #         print("The target has been reached, stopping the simulation")
                    #         break
                    # if args.agent == "LBC":
                    #     observations = world.get_observations()
                    #     control = agent.run_step(observations)
                    # else:
                    observations = world.get_observations(agent)
                    # print(observations.keys())
                    control = agent.run_step()
                    # control = agent.run_step()
                    control.manual_gear_shift = False
                    world.player.apply_control(control)
                processed = process(observations)
                
                data.append(processed)
                # print(data[-1]['measurements']['command'], len(data))
                progress.update(1)
            progress.close()

            lmdb_env = lmdb.open(str(filepath), map_size=int(1e10))
            n = len(data)

            with lmdb_env.begin(write=True) as txn:
                txn.put('len'.encode(), str(n).encode())
                for i, x in enumerate(data):
                    txn.put(('rgb_left_%04d' % i).encode(), np.ascontiguousarray(x['rgb_left']).astype(np.uint8))
                    txn.put(('rgb_right_%04d' % i).encode(), np.ascontiguousarray(x['rgb_right']).astype(np.uint8))
                    txn.put(('birdview_%04d' % i).encode(), np.ascontiguousarray(x['birdview']).astype(np.uint8))
                    txn.put(('measurements_%04d' % i).encode(), np.ascontiguousarray(x['measurements']).astype(np.float32))
                    txn.put(('control_%04d' % i).encode(), np.ascontiguousarray(x['control']).astype(np.float32))

            if world is not None:
                settings = world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.world.apply_settings(settings)
                # traffic_manager.set_synchronous_mode(True)

                world.destroy()

                

    finally:

        if world is not None:
            settings = world.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            world.world.apply_settings(settings)
            # traffic_manager.set_synchronous_mode(True)

            world.destroy()

        pygame.quit()


# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================


def main():
    """Main method"""

    argparser = argparse.ArgumentParser(
        description='CARLA Automatic Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='Actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')
    argparser.add_argument(
        "-a", "--agent", type=str,
        choices=["Behavior", "Basic", "LBC"],
        help="select which agent to run",
        default="Behavior")
    argparser.add_argument(
        '-b', '--behavior', type=str,
        choices=["cautious", "normal", "aggressive"],
        help='Choose one of the possible agent behaviors (default: cautious) ',
        default='normal')
    argparser.add_argument(
        '-s', '--seed',
        help='Set seed for repeating executions (default: None)',
        default=None,
        type=int)
    argparser.add_argument(
        '--n-vehicles',
        help='Set number of vehicles in the simulation (default: 0)',
        default=50,
        type=int)
    argparser.add_argument(
        '--n-pedestrians',
        help='Set number of pedestrians in the simulation (default: 0)',
        default=100,
        type=int)
    argparser.add_argument(
        '--model-path',
        help='Path of the model',
        default='/home/dse_c/carla_lbc/ckpts/image/model-10.th')
    argparser.add_argument(
        '--town',
        help='Set the town for rendering',
        default='Town01')
    argparser.add_argument(
        '--weather',
        help='Set the weather for rendering',
        default=1,
        type=int)
    argparser.add_argument(
        '--frames-per-episode',
        help='Set the weather for rendering',
        default=1000,
        type=int)
    argparser.add_argument(
        '--frames_skip',
        help='Set the weather for rendering',
        default=1,
        type=int)
    argparser.add_argument(
        '--dataset-path',
        help='Set the weather for rendering',
        default='/media/storage/karthik/lbc/data_vehicles')
    argparser.add_argument(
        '--n-episodes',
        help='Set the weather for rendering',
        default=2,
        type=int)
    argparser.add_argument(
        '--debug-waypoints',
        help='Set the weather for rendering',
        action='store_true',
        dest='debug_waypoints')
    

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:
        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    main()
