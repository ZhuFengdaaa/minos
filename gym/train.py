#!/usr/bin/env python3

import os
import time
import numpy as np
import argparse
import gym
import gym_minos
import matplotlib.pyplot as plt
import torch
import torch.multiprocessing as mp
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from itertools import count
from torch.autograd import Variable
import torch.nn.functional as F
import pretrainedmodels
import torch.optim as optim
from PIL import Image

from minos.config.sim_args import parse_sim_args
from torch.distributions import Categorical

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2i = nn.Linear(input_size + hidden_size, output_size)
        self.i2c = nn.Linear(input_size + hidden_size, output_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)

    def forward(self, input, hidden):
        combined = torch.cat([input, hidden], dim=1)
        f_gate = F.sigmoid(self.i2f(combined))
        i_gate = F.sigmoid(self.i2i(combined))
        o_gate = F.sigmoid(self.i2o(combined))
        _hidden = F.tanh(self.i2c(combined))
        hidden = f_gate * hidden + i_gate * _hidden
        output = o_gate * F.tanh(hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

class BaseModel(nn.Module):
    def __init__(self, gpu_id):
        super(BaseModel, self).__init__()
        # self.cnn = models.resnet18(pretrained=True)
        self.gpu_id = gpu_id
        self.cnn = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')

        self.avg_pool = nn.AvgPool2d(7, stride=1)
        self.affine1 = nn.Linear(1024, 512)
        self.rnn = RNN(512, 512, 512)
        self.affine2 = nn.Linear(512, 512)
        self.action_head = nn.Linear(512, 3) # turn left, right, forward
        self.value_head = nn.Linear(512, 1)
        if gpu_id==-1:
            self.hidden = Variable(torch.zeros(1, 512))
        else:
            self.hidden = Variable(torch.zeros(1, 512)).cuda(gpu_id)

    def forward(self, c, d):
        c = self.cnn.features(c) # 3584 x 7
        c = self.avg_pool(c)
        c = c.view(c.size(0), -1)
        d = self.cnn.features(d) # 3584 x 7
        d = self.avg_pool(d)
        d = d.view(d.size(0), -1)
        x = torch.cat([c,d], dim=-1)
        x = F.relu(self.affine1(x))
        x = self.rnn(x, self.hidden)[0]
        x = F.relu(self.affine2(x))
        action_scores = self.action_head(x)
        state_scores = self.value_head(x)
        return action_scores, state_scores

    def do_train(self):
        self.train()
        self.cnn.eval()

    def do_eval(self):
        self.eval()

def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(),
                                   shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
        
actions = np.eye(3)
action_texts = ["left", "right", "forward"]

def train(rank, args, shared_model):
    if rank == 0:
        checkpoint_dir = os.path.join(args.checkpoint_dir, time.strftime("%Y_%m_%d_%H_%M_%S"))
    args.seed = args.seed + rank * 127
    gpu_id = args.gpus[rank % len(args.gpus)]
    torch.cuda.set_device(gpu_id)
    env = gym.make('indoor-v0')
    torch.manual_seed(args.seed)
    eps = np.finfo(np.float32).eps.item()
    max_iter = 100
    optimizer = optim.Adam(shared_model.parameters(), lr=args.lr)
    
    model = BaseModel(gpu_id)
    model.do_train()
    env.configure(args)
    running_reward = 10
    for i_episode in count(1):
        model.load_state_dict(shared_model.state_dict())
        model.cuda()
        state = env.reset()
        values = []
        log_probs = []
        rewards = []
        entropies = []
        for t in range(max_iter):
            if rank==0 and args.render:
                env.render()
            if args.save_observations: 
                pass
            # print(type(state['observation']['sensors']['color']['data']))
            color = state['observation']['sensors']['color']['data']
            depth = state['observation']['sensors']['depth']['data']
            color = Image.fromarray(color).convert('RGB')
            depth = Image.fromarray(depth).convert('RGB')
            color = Variable(transform(color)).cuda()
            depth = Variable(transform(depth)).cuda()
            logit, value = model(color.unsqueeze(0), depth.unsqueeze(0))
            prob = F.softmax(logit, dim=-1)
            log_prob = F.log_softmax(logit, dim=-1)
            entropy = -(log_prob * prob).sum(1, keepdim=True)
            entropies.append(entropy)

            action = prob.multinomial(num_samples=1).data
            log_prob = log_prob.gather(1, Variable(action))

            action = action.cpu().numpy().item()
            state, reward, done, _ = env.step(actions[action])
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            R = value.clone()
            if done:
                break
        if rank==0 and args.render:
            env.render()

        print("total rewards: ", sum(rewards), " total steps: ", len(rewards))
        values.append(R)
        policy_loss = 0
        value_loss = 0
        gae = torch.zeros(1, 1).cuda()
        for i in reversed(range(len(rewards))):
            R = args.gamma * R + rewards[i]
            advantage = R - values[i]
            value_loss = value_loss + 0.5 * advantage.pow(2)

            # Generalized Advantage Estimataion
            delta_t = rewards[i] + args.gamma * \
                values[i + 1].data - values[i].data
            gae = gae * args.gamma * args.tau + delta_t.cuda()

            policy_loss = policy_loss - \
                log_probs[i] * Variable(gae) - args.entropy_coef * entropies[i]
        
        optimizer.zero_grad()
        loss = policy_loss + args.value_loss_coef * value_loss
        print("loss: ", loss.data)
        loss.backward()
        # TODO: test if necessary
        if args.clip_grad:
            torch.nn.utils.clip_grad_norm(model.parameters(), args.max_grad_norm)
        ensure_shared_grads(model.cpu(), shared_model)
        optimizer.step()

        if i_episode % args.log_interval == 0:
            print('Rank {}: Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                rank, i_episode, t, running_reward))
            print("Rank {}: Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(rank, running_reward, t))
        
        if rank == 0 and i_episode % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, "{}_{}.pt".format(rank, i_episode))
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            torch.save(shared_model, checkpoint_path)

def run_gym(args):
    print('Running MINOS gym example')
    print("CUDA_VISIBLE_DEVICES=" + os.environ['CUDA_VISIBLE_DEVICES'])
    args.gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    args.gpus = [int(x) for x in args.gpus]

    shared_model = BaseModel(-1)
    shared_model.share_memory()
    if args.load_path != "":
        print('Loading params from checkpoint: %s' % args.load_path)
        shared_model.load_state_dict(torch.load(args.load_path).state_dict())

    processes = []
    for rank in range(args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()


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
    parser.add_argument('--load_path', default="", help="/home/linchao/minos/gym/checkpoint/2018_09_06_22_41_31/0_600.pt")
    parser.add_argument('--checkpoint_dir', default="./checkpoint")
    parser.add_argument('--num_processes', type=int, default=1)
    parser.add_argument('--seed', default=0)
    parser.add_argument('--render', action='store_true', default=False)
    parser.add_argument('--gamma', default=0.99)
    parser.add_argument('--tau', type=float, default=1.00)
    parser.add_argument('--entropy-coef', type=float, default=0.01)
    parser.add_argument('--value-loss-coef', type=float, default=0.5)
    parser.add_argument('--clip_grad', type=bool, default=True)
    parser.add_argument('--max_grad_norm', type=float, default=50)
    parser.add_argument('--log_interval', default=10)
    parser.add_argument('--save_interval', default=10)
    parser.add_argument('--lr', type=float, default=0.0001)
    args = parse_sim_args(parser)
    run_gym(args)


if __name__ == "__main__":
    main()
