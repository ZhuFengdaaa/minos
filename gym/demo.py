#!/usr/bin/env python3

import argparse
import gym
import gym_minos
import time
import os
import numpy as np
import matplotlib.pyplot as plt

from minos.config.sim_args import parse_sim_args

def run_gym(sim_args):
    start = time.time()
    image_id = 0
    save_path = 'datasets/{}'.format(sim_args.source)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    env = gym.make('indoor-v0')
    env.configure(sim_args)
    while image_id != 50000:
        observation = env.reset()
        image_id = save_observations(observation, sim_args, save_path, image_id)
        image_id += 1
        if image_id % 200 == 0:
            print("Use {:.3f}s save {} images; ".format(time.time() - start, image_id))


    # start = time.time()
    # while image_id != 10000:
    #     env = gym.make('indoor-v0')
    #     env.configure(sim_args)
    #     # print('Running MINOS gym example')
    #     for i_episode in range(10):
    #         # print('Starting episode %d' % i_episode)
    #         observation = env.reset()
    #         done = False
    #         num_steps = 0
    #         while not done:
    #             # env.render()
    #             action = env.action_space.sample()
    #             observation, reward, done, info = env.step(action)
    #             if sim_args.save_observations and num_steps % 20 == 0:
    #                 image_id = save_observations(observation, sim_args, save_path, image_id)
    #                 image_id += 1
    #             num_steps += 1
    #         print("Episode finished after {} steps;  Use {}s save {} images; ".format(i_episode,
    #                                                                                   time.time() - start,
    #                                                                                   image_id))


def save_observations(observation, sim_args, save_path, image_id):
    if sim_args.observations.get('color'):
        color = observation["observation"]["sensors"]["color"]["data"]
        if color.sum() < 100:
            print('Bad image')
            image_id -= 1
        else:
            plt.imsave('{}/{}.png'.format(save_path, image_id), color)
        return image_id

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

    # shortest_path = observation["observation"]["measurements"]["shortest_path_to_goal"]
    # print(shortest_path)


def main():
    parser = argparse.ArgumentParser(description='MINOS gym wrapper')
    parser.add_argument('--save_observations', action='store_true',
                        default=False,
                        help='Save sensor observations at each step to images')
    args = parse_sim_args(parser)
    # print(args)
    # assert False
    run_gym(args)
    # scans_list = os.listdir('/home/xuguanghui/work/mp3d/scans')
    # # scans_list = os.listdir('/home/xuguanghui/work/suncg/navgrid_10')
    # for scans in scans_list:
    #     args.scene_ids = scans
    #     run_gym(args)


if __name__ == "__main__":
    main()
