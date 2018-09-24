#!/usr/bin/env python3

import argparse
import gym
import gym_minos
import matplotlib.pyplot as plt
import time
import random
from PIL import Image
import os
import json

from minos.config.sim_args import parse_sim_args

action_pool = [[0,0,1], [1, 0, 0], [0, 1, 0], [1, 0, 1], [0, 1, 1]]

episode_info={}
def run_gym(sim_args):
    env = gym.make('indoor-v0')
    env.configure(sim_args)
    print('Running MINOS gym example')
    episode_count=0
    for i_episode in range(10000):
        print('Starting episode %d' % i_episode)
        observation = env.reset()
        _episode_info = observation[2]
        done = False
        num_steps = 0
        action = action_pool[int(random.random()*len(action_pool))]
        print(action)
        colors = []
        depths = []
        forces = []
        pos = []
        ori = []
        cnt=0
        flag=True
        for _ in range(50):
            observation, reward, done, info = env.step(action)
            colors.append(observation[0][:,:,:4])
            depths.append(observation[0][:,:,4])
            forces.append(observation[1][9:])
            pos.append(observation[2])
            ori.append(observation[3])
            if len(pos)>1 and pos[-1]==pos[-2] and ori[-1]==ori[-2]:
                cnt+=1
            else:
                cnt=0
            if cnt>=5:
                flag=False
                break
        if flag==True:
            print("Saving episode {}".format(i_episode))
            directory = "./{}".format(episode_count)
            os.mkdir(directory)
            for i, (color, depth) in enumerate(zip(colors, depths)):
                Image.fromarray(color.astype('uint8')).convert('RGB').save(os.path.join(directory, 'color{}.png'.format(i)))
                Image.fromarray(depth, 'L').save(os.path.join(directory, 'depth{}.png'.format(i)))
            episode_info[episode_count]=_episode_info
            episode_count+=1
    with open("episode_info.txt", "w") as outfile:
        json.dump(episode_info, outfile, indent=4)

def save_observations(observation, sim_args):
    if sim_args.observations.get('color'):
        color = observation["observation"]["sensors"]["color"]["data"]
        plt.imsave('color.png', color)

    if sim_args.observations.get('depth'):
        depth = observation["observation"]["sensors"]["depth"]["data"]
        plt.imsave('depth.png', depth, cmap='Greys')

    if sim_args.observations.get('normal'):
        normal = observation["observation"]["sensors"]["normal"]["data"]
        plt.imsave('normal.png', normal)

    if sim_args.observations.get('objectId'):
        object_id = observation["observation"]["sensors"]["objectId"]["data"]
        plt.imsave('object_id.png', object_id)

    if sim_args.observations.get('objectType'):
        object_type = observation["observation"]["sensors"]["objectType"]["data"]
        plt.imsave('object_type.png', object_type)

    if sim_args.observations.get('roomId'):
        room_id = observation["observation"]["sensors"]["roomId"]["data"]
        plt.imsave('room_id.png', room_id)

    if sim_args.observations.get('roomType'):
        room_type = observation["observation"]["sensors"]["roomType"]["data"]
        plt.imsave('room_type.png', room_type)

    if sim_args.observations.get('map'):
        nav_map = observation["observation"]["map"]["data"]
        nav_map.shape = (nav_map.shape[1], nav_map.shape[0], nav_map.shape[2])
        plt.imsave('nav_map.png', nav_map)

    shortest_path = observation["observation"]["measurements"]["shortest_path_to_goal"]
    print(shortest_path)


def main():
    parser = argparse.ArgumentParser(description='MINOS gym wrapper')
    parser.add_argument('--save_observations', action='store_true',
                        default=False,
                        help='Save sensor observations at each step to images')
    args = parse_sim_args(parser)
    run_gym(args)


if __name__ == "__main__":
    main()
