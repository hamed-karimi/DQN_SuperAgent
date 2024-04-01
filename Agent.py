import os
import random
import torch
import numpy as np
from torch.nn import init
from torch import nn, optim
from DQN import DQN, weights_init_orthogonal
from gymnasium import spaces
import math
from ReplayMemory import ReplayMemory


class Agent:
    def __init__(self, params, device='auto'):
        self.params = params
        self.device = 'cuda' if ((device == 'auto' or device == 'cuda') and torch.cuda.is_available()) else 'cpu'
        self.gamma = params.GAMMA
        self.policy_net = DQN(params, self.device)
        self.target_net = DQN(params, self.device)
        self.policy_net.apply(weights_init_orthogonal)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.epsilon = .95
        self.epsilon_range = [.95, .05]
        self.batch_size = params.BATCH_SIZE
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=params.INIT_LEARNING_RATE)
        self.memory = ReplayMemory(capacity=self.params.MEMORY_CAPACITY)

    def get_policy_net_goal_map(self, state):
        env_map = np.array(state[0])
        mental_state = torch.Tensor(state[1]).unsqueeze(dim=0)
        mental_state_slope = torch.Tensor(state[2]).unsqueeze(dim=0)
        object_reward = torch.Tensor(state[3]).unsqueeze(dim=0)
        with torch.no_grad():
            goal_values = self.policy_net(env_map=torch.Tensor(env_map).unsqueeze(dim=0),
                                          mental_states=mental_state,
                                          states_params=torch.concat([mental_state_slope, object_reward], dim=1))
            goal_values = goal_values.cpu()
        goal_location = self._get_goal_location_from_values(values=goal_values, env_map=torch.tensor(env_map))

        return goal_location

    def get_action(self, state: list, episode, epsilon=None):
        env_map = np.array(state[0])
        goal_map = np.zeros_like(env_map[0, :, :])
        epsilon = self.epsilon if epsilon is None else epsilon
        if random.random() < epsilon:  # random action
            all_object_locations = np.stack(np.where(env_map), axis=1)
            goal_index = np.random.randint(low=0, high=all_object_locations.shape[0], size=())
            goal_location = all_object_locations[goal_index, 1:]

        else:
            goal_location = self.get_policy_net_goal_map(state)

        goal_map[goal_location[0], goal_location[1]] = 1
        self._update_epsilon(episode=episode)
        return goal_map

    def _update_epsilon(self, episode):
        epsilon_length = self.epsilon_range[0] - self.epsilon_range[1]
        self.epsilon = self.epsilon_range[0] - (episode / self.params.EPISODE_NUM) * epsilon_length

    def _get_goal_location_from_values(self, values, env_map: torch.Tensor):
        goal_values = values.reshape(self.params.HEIGHT, self.params.WIDTH).clone()
        object_mask = env_map.sum(dim=0) > 0
        goal_values[~object_mask] = -math.inf
        goal_location = np.array(np.unravel_index(goal_values.argmax(), goal_values.shape))
        return goal_location

    def _get_batch_tensor(self, batch):
        # 'init_map', 'init_mental_state', 'states_params',
                                      # 'goal_map', 'reward',
                                      # 'next_map', 'next_mental_state'
        assert type(batch) is self.memory.Transition, 'batch should be a memory Transition name tuple'
        init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state = [], [], [], [], [], [], []
        for i in range(len(batch.init_map)):
            init_map.append(batch.init_map[i])
            init_mental_state.append(batch.init_mental_state[i])
            states_params.append(batch.states_params[i])
            goal_map.append(batch.goal_map[i])
            reward.append(batch.reward[i])
            next_map.append(batch.next_map[i])
            next_mental_state.append(batch.next_mental_state[i])
        init_map = torch.stack(init_map, dim=0)
        init_mental_state = torch.stack(init_mental_state, dim=0)
        states_params = torch.stack(states_params, dim=0)
        goal_map = torch.stack(goal_map, dim=0)
        reward = torch.stack(reward, dim=0)
        next_map = torch.stack(next_map, dim=0)
        next_mental_state = torch.stack(next_mental_state, dim=0)
        return init_map, init_mental_state, states_params, goal_map, reward, next_map, next_mental_state

    def save_experience(self, *args):
        self.memory.push_experience(*args)

    def optimize(self):
        if len(self.memory) < 3 * self.batch_size:
            return 0.
        transition_sample = self.memory.sample(self.batch_size)
        batch = self.memory.Transition(*zip(*transition_sample))
        self.policy_net.train()
        # ('init_state', 'goal_map', 'reward', 'next_state')
        init_map, \
            init_mental_state, \
            states_params, \
            goal_map, \
            reward, \
            next_map, \
            next_mental_state = self._get_batch_tensor(batch)

        policy_net_goal_values = self.policy_net(init_map, init_mental_state, states_params).cpu()
        policy_net_goal_values = policy_net_goal_values[goal_map > 0]

        next_state_target_net_goal_values = self.target_net(next_map, next_mental_state, states_params).cpu()
        next_state_target_net_goal_values[next_map.sum(dim=1) < 1] = -math.inf
        target_net_max_goal_value = torch.amax(next_state_target_net_goal_values,
                                               dim=(1, 2)).detach().float()

        expected_goal_values = reward + target_net_max_goal_value * self.gamma

        criterion = nn.SmoothL1Loss()
        loss = criterion(policy_net_goal_values, expected_goal_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        return loss

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), os.path.join(path, 'model.pt'))
