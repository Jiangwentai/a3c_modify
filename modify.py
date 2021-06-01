import os
import argparse
import numpy as np
import torch.multiprocessing as _mp
import shutil
import torch
from torch.distributions import Categorical
from collections import deque
import timeit
import torch.nn as nn
import torch.nn.functional as F
from breakout_wrap import make_atari,wrap_deepmind

os.environ['OMP_NUM_THREADS'] = '2'

SHAPE=[1,4 ,84, 84]

class GlobalAdam(torch.optim.Adam):
    def __init__(self, params, lr):
        super(GlobalAdam, self).__init__(params, lr=lr)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()


class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 8, stride=4, padding=1)

        self.conv2 = nn.Conv2d(16, 32, 4, stride=2, padding=1)
        self.fc=nn.Linear(32*10*10,256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, 2)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                # nn.init.kaiming_uniform_(module.weight)
                # nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.fc(x.view(x.size(0), -1))

        return self.actor_linear(x), self.critic_linear(x)


def local_train(lock,index, opt, global_model, optimizer, save=False):
    torch.manual_seed(123 + index)
    if save:
        start_time = timeit.default_timer()
    env = make_atari("PongNoFrameskip-v4")
    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)
    local_model = ActorCritic()
    if opt.use_gpu:
        local_model.cuda()
    local_model.train()
    state = torch.from_numpy(np.array(env.reset()).reshape(SHAPE))
    if opt.use_gpu:
        state = state.cuda()
    done = True
    curr_step = 0
    curr_episode = 0
    while True:

        curr_episode += 1
        local_model.load_state_dict(global_model.state_dict())

        log_policies = []
        values = []
        rewards = []
        entropies = []

        for _ in range(opt.num_local_steps):
            curr_step += 1
            logits, value = local_model(state)
            policy = F.softmax(logits, dim=1)
            log_policy = F.log_softmax(logits, dim=1)
            entropy = -(policy * log_policy).sum(1, keepdim=True)

            m = Categorical(policy)
            action = m.sample().item()

            state, reward, done, _ = env.step(action+2)

            state = torch.from_numpy(np.array(state).reshape(SHAPE))

            if opt.use_gpu:
                state = state.cuda()
            if curr_step > opt.num_global_steps:
                done = True

            if done:
                curr_step = 0
                state = torch.from_numpy(np.array(env.reset()).reshape(SHAPE))
                if opt.use_gpu:
                    state = state.cuda()

            values.append(value)
            log_policies.append(log_policy[0, action])
            rewards.append(reward)
            entropies.append(entropy)

            if done:
                break

        R = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            R = R.cuda()
        if not done:
            _, R = local_model(state)

        gae = torch.zeros((1, 1), dtype=torch.float)
        if opt.use_gpu:
            gae = gae.cuda()
        actor_loss = 0
        critic_loss = 0
        entropy_loss = 0
        next_value = R

        for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
            gae = gae * opt.gamma * opt.tau
            gae = gae + reward + opt.gamma * next_value.detach() - value.detach()
            next_value = value
            actor_loss = actor_loss + log_policy * gae
            R = R * opt.gamma + reward
            critic_loss = critic_loss + (R - value) ** 2 / 2
            entropy_loss = entropy_loss + entropy

        total_loss = -actor_loss + critic_loss - opt.beta * entropy_loss
        optimizer.zero_grad()
        total_loss.backward()

        for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
            if global_param.grad is not None:
                print(global_param.grad)
                break
            global_param._grad = local_param.grad
            print(global_param.grad)

        optimizer.step()

        if curr_episode == int(opt.num_global_steps / opt.num_local_steps):
            print("Training process {} terminated".format(index))
            if save:
                end_time = timeit.default_timer()
                print('The code runs for %.2f s ' % (end_time - start_time))
            return


def local_test(index, opt, global_model):
    torch.manual_seed(123 + index)
    env = make_atari("PongNoFrameskip-v4")

    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)
    local_model = ActorCritic()
    local_model.eval()
    state = torch.from_numpy(np.array(env.reset()).reshape(SHAPE))
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())

        logits, value = local_model(state)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        state, reward, done, _ = env.step(action+2)
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(np.array(state).reshape(SHAPE))


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of model described in the paper: Asynchronous Methods for Deep Reinforcement Learning for Super Mario Bros""")
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="complex")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards')
    parser.add_argument('--tau', type=float, default=1.0, help='parameter for GAE')
    parser.add_argument('--beta', type=float, default=0.01, help='entropy coefficient')
    parser.add_argument("--num_local_steps", type=int, default=5)
    parser.add_argument("--num_global_steps", type=int, default=5e6)
    parser.add_argument("--num_processes", type=int, default=6)
    parser.add_argument("--save_interval", type=int, default=500, help="Number of steps between savings")
    parser.add_argument("--max_actions", type=int, default=200, help="Maximum repetition steps in test phase")
    parser.add_argument("--log_path", type=str, default="tensorboard/a3c_super_mario_bros")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--load_from_previous_stage", type=bool, default=False,
                        help="Load weight from previous trained stage")
    parser.add_argument("--use_gpu", type=bool, default=False)
    args = parser.parse_args()
    return args


def train(opt):
    torch.manual_seed(123)
    if os.path.isdir(opt.log_path):
        shutil.rmtree(opt.log_path)
    os.makedirs(opt.log_path)
    if not os.path.isdir(opt.saved_path):
        os.makedirs(opt.saved_path)
    mp = _mp.get_context("spawn")
    env = make_atari("PongNoFrameskip-v0")
    env = wrap_deepmind(env, frame_stack=True, scale=True, episode_life=True)
    global_model = ActorCritic()
    if opt.use_gpu:
        global_model.cuda()
    global_model.share_memory()

    lock=mp.Manager().Lock()
    optimizer = GlobalAdam(global_model.parameters(), lr=opt.lr)
    processes = []
    for index in range(opt.num_processes):
        if index == 0:
            process = mp.Process(target=local_train, args=(lock,index, opt, global_model, optimizer, True))
        else:
            process = mp.Process(target=local_train, args=(lock,index, opt, global_model, optimizer))
        process.start()
        processes.append(process)
    process = mp.Process(target=local_test, args=(opt.num_processes, opt, global_model))
    process.start()
    processes.append(process)
    for process in processes:
        process.join()


if __name__ == "__main__":
    opt = get_args()
    train(opt)
