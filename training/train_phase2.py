import time
import argparse

from pathlib import Path

import numpy as np
import tqdm

import glob
import os
import sys


try:
    sys.path.append(glob.glob('../PythonAPI')[0])
    sys.path.append(glob.glob('../bird_view')[0])
    sys.path.append(glob.glob('../')[0])
except IndexError as e:
    pass

from bird_view.utils import carla_utils as cu
from train_util import one_hot
from benchmark import make_suite
from bird_view.models import image, birdview
from bird_view.models.image import ImagePolicyModelSS, ImageAgent
from bird_view.models.birdview import BirdViewPolicyModelSS, BirdViewAgent

BACKBONE = 'resnet34'
GAP = 5
N_STEP = 5
CROP_SIZE = 192
MAP_SIZE = 320
SAVE_EPISODES = list(range(20))


from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

def crop_birdview(birdview, dx=0, dy=0):
    x = 260 - CROP_SIZE // 2 + dx
    y = MAP_SIZE // 2 + dy

    birdview = birdview[
            x-CROP_SIZE//2:x+CROP_SIZE//2,
            y-CROP_SIZE//2:y+CROP_SIZE//2]

    return birdview

        
def get_control(agent_control, teacher_control, episode, beta=0.95):
    """
    A learning schedule to choose between agent control and teacher control
    lim_{episode->inf} P(control=agent_control|episode) = 1
    to make sure it converges
    """
    prob = 0.5 + 0.5*(1-beta**episode)

    if np.random.uniform(0,1) < prob:
        control = agent_control
    else:
        control = teacher_control
        
    return control



def rollout(replay_buffer, coord_converter, net, teacher_net, episode, 
        image_agent_kwargs=dict(), birdview_agent_kwargs=dict(),
        episode_length=1000,
        n_vehicles=100, n_pedestrians=250, port=2000, planner="new"):
    
    decay = np.array([0.7**i for i in range(5)])
    xy_bias = np.array([0.7,0.3])

    weathers = list(cu.TRAIN_WEATHERS.keys())
        
    def _get_weight(a, b):
        loss_weight = np.mean((np.abs(a - b)*xy_bias).sum(axis=-1)*decay, axis=-1)
        x_weight = np.maximum(
            np.mean(a[...,0],axis=-1),
            np.mean(a[...,0]*-1.4,axis=-1),
        )
        
        return loss_weight

    num_data = 0
    progress = tqdm.tqdm(range(episode_length), desc='Frame')
    
    
    data = list()

    while len(data) < episode_length:

        try:
            # Start Carla Client
            client = carla.Client('localhost', 2000)
            client.set_timeout(4.0)
            sim_world = client.load_world('Town01')

            display = pygame.display.set_mode(
            (args.width, args.height),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
            display2 = None
            hud = HUD(args.width, args.height)
            world = World(carla_world=client.get_world(), hud=hud, args=args, client=client, display2=display2)
            controller = KeyboardControl(world)
            image_agent = ImageAgent(world.player, model=net)
            birdview_agent = BirdViewAgent(model=teacher_net)

            # Set the agent destination
            spawn_points = world.map.get_spawn_points()
            destination = random.choice(spawn_points).location
            image_agent.set_destination(destination)

            clock = pygame.time.Clock()
            while True:
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

                if image_agent.done():
                    if args.loop:
                        image_agent.set_destination(random.choice(spawn_points).location)
                        world.hud.notification("The target has been reached, searching for another target", seconds=4.0)
                        print("The target has been reached, searching for another target")
                    else:
                        print("The target has been reached, stopping the simulation")
                        break
                observations = world.get_observations(image_agent)
                image_control, _image_points = image_agent.run_step(observations, teaching=True)
                _image_points = coord_converter(_image_points)
                birdview_control, birdview_points = birdview_agent.run_step(observations, teaching=True)
                weight = _get_weight(birdview_points, image_points)
                control = get_control(image_control, birdview_control, episode)
                control.manual_gear_shift = False
                world.player.apply_control(control)
    
                data.append({
                    'rgb_img_left': observations["rgb_left"].copy(),
                    'rgb_img_right': observations["rgb_right"].copy(),
                    'cmd': int(observations["command"]),
                    'speed': np.linalg.norm(observations["velocity"]),
                    'target': birdview_points,
                    'weight': weight,
                    'birdview_img': crop_birdview(observations['birdview'], dx=-10),
                })
                
                progress.update(1)

                if len(data) >= episode_length:
                    break
                


        finally:

            if world is not None:
                settings = world.world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                world.world.apply_settings(settings)

                world.destroy()

            pygame.quit()
            
    for datum in data:
        replay_buffer.add_data(**datum)
        num_data += 1


def _train(replay_buffer, net, teacher_net, criterion, coord_converter, logger, config, episode):
    
    import torch
    from phase2_utils import _log_visuals, get_weight, repeat

    teacher_net.eval()

    for epoch in range(config['epoch_per_episode']):
                     
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)
        
        net.train()
        replay_buffer.init_new_weights()
        loader = torch.utils.data.DataLoader(replay_buffer, batch_size=config['batch_size'], num_workers=4, shuffle=True, drop_last=True)

        for i, (idxes, rgb_image, command, speed, target, birdview) in enumerate(loader):
            if i % 100 == 0:
                print ("ITER: %d"%i)
            rgb_image_left = rgb_image_left.to(config['device']).float()
            rgb_image_right = rgb_image_right.to(config['device']).float()
            birdview = birdview.to(config['device']).float()
            command = one_hot(command).to(config['device']).float()
            speed = speed.to(config['device']).float()
            
            if len(rgb_image.size()) > 4:
                B, batch_aug, c, h, w = rgb_image.size()
                rgb_image_left = rgb_image_left.view(B*batch_aug,c,h,w)
                rgb_image_right = rgb_image_right.view(B*batch_aug,c,h,w)
                birdview = repeat(birdview, batch_aug)
                command = repeat(command, batch_aug)
                speed = repeat(speed, batch_aug)
            else:
                B = rgb_image.size(0)
                batch_aug = 1
            
            with torch.no_grad():
                _teac_location, _teac_locations = teacher_net(birdview, speed, command)
                
            _pred_location, _pred_locations = net(rgb_image_left, rgb_image_right, speed, command)
            pred_location = coord_converter(_pred_location)
            pred_locations = coord_converter(_pred_locations)
            
            optimizer.zero_grad()
            loss = criterion(pred_locations, _teac_locations)
            
            # Compute resample weights
            pred_location_normed = pred_location / (0.5*CROP_SIZE) - 1.
            weights = get_weight(pred_location_normed, _teac_location)
            weights = torch.mean(torch.stack(torch.chunk(weights,B)),dim=0)
            
            
            replay_buffer.update_weights(idxes, weights)

            loss_mean = loss.mean()
            
            loss_mean.backward()
            optimizer.step()
            
            loss_.append(loss_mean.item())
        
        replay_buffer.normalize_weights()

        rgb_image_left, rgb_image_right, birdview, command, speed, target = replay_buffer.get_highest_k(32)
        rgb_image_left = rgb_image_left.to(config['device']).float()
        rgb_image_right = rgb_image_right.to(config['device']).float()
        birdview = birdview.to(config['device']).float()
        command = one_hot(command).to(config['device']).float()
        speed = speed.to(config['device']).float()
        
        with torch.no_grad():
            _teac_location, _teac_locations = teacher_net(birdview, speed, command)
    
        net.eval()
        _pred_location, _pred_locations = net(rgb_image_left, rgb_image_right, speed, command)
        pred_location = coord_converter(_pred_location)
        pred_locations = coord_converter(_pred_locations)
        pred_location_normed = pred_location / (0.5*CROP_SIZE) - 1.
        weights = get_weight(pred_location_normed, _teac_location)
        

    if episode in SAVE_EPISODES:
        torch.save(net.state_dict(),
            str(Path(config['log_dir']) / ('model-%d.th' % episode)))


def train(config):

    import utils.bz_utils as bzu

    # bzu.log.init(config['log_dir'])
    # bzu.log.save_config(config)
    # teacher_config = bzu.log.load_config(config['teacher_args']['model_path'])
    
    from phase2_utils import (
        CoordConverter, 
        ReplayBuffer, 
        LocationLoss, 
        load_birdview_model,
        load_image_model,
        get_optimizer
        )
    
    criterion = LocationLoss()
    net = load_image_model(
        config['model_args']['backbone'], 
        config['phase1_ckpt'],
        device=config['device'])
        
    teacher_net = load_birdview_model(
        teacher_config['model_args']['backbone'], 
        config['teacher_args']['model_path'], 
        device=config['device'])
        
    image_agent_kwargs = { 'camera_args' : config["agent_args"]['camera_args'] }

    coord_converter = CoordConverter(**config["agent_args"]['camera_args'])
        
    replay_buffer = ReplayBuffer(**config["buffer_args"])
        
    # optimizer = get_optimizer(net.parameters(), config["optimizer_args"]["lr"])

    for episode in tqdm.tqdm(range(config['max_episode']), desc='Episode'):
        rollout(replay_buffer, coord_converter, net, teacher_net, episode, image_agent_kwargs=image_agent_kwargs, port=config['port'])
        # import pdb; pdb.set_trace()
        _train(replay_buffer, net, teacher_net, criterion, coord_converter, bzu.log, config, episode)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', default='/media/storage/karthik/lbc')
    parser.add_argument('--log_iterations', default=100)
    parser.add_argument('--max_episode', default=20)
    parser.add_argument('--epoch_per_episode', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--speed_noise', type=float, default=0.0)
    parser.add_argument('--batch_aug', type=int, default=1)

    parser.add_argument('--ckpt', required=True)
    
    # Teacher.
    parser.add_argument('--teacher_path', default='/media/storage/karthik/lbc/ckpts/priveleged/model-128.th')
    
    parser.add_argument('--fixed_offset', type=float, default=4.0)
    
    # Optimizer.
    parser.add_argument('--lr', type=float, default=1e-4)
    
    # Misc
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument(
        '-l', '--loop',
        action='store_true',
        dest='loop',
        help='Sets a new random destination upon reaching the previous one (default: False)')

    parser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='Window resolution (default: 1280x720)')
    parser.add_argument(
        '--model-path',
        help='Path of the model',
        default='/media/storage/karthik/lbc/checkpoints/phase1/model-800.th')


    
    parsed = parser.parse_args()
    
    
    config = {
            'port': parsed.port,
            'log_dir': parsed.log_dir,
            'log_iterations': parsed.log_iterations,
            'batch_size': parsed.batch_size,
            'max_episode': parsed.max_episode,
            'speed_noise': parsed.speed_noise,
            'epoch_per_episode': parsed.epoch_per_episode,
            'device': 'cuda',
            'phase1_ckpt': parsed.ckpt,
            'optimizer_args': {'lr': parsed.lr},
            'buffer_args': {
                'buffer_limit': 200000,
                'batch_aug': parsed.batch_aug,
                'augment': 'super_hard',
                'aug_fix_iter': 819200,
            },
            'model_args': {
                'model': 'image_ss',
                'backbone': BACKBONE,
                },
            'agent_args': {
                'camera_args': {
                    'w': 384,
                    'h': 160,
                    'fov': 90,
                    'world_y': 1.4,
                    'fixed_offset': parsed.fixed_offset,
                }
            },
            'teacher_args' : {
                'model_path': parsed.teacher_path,
            }
        }

    train(config)
