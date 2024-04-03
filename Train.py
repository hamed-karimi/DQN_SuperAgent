import os.path
import shutil
from copy import deepcopy
from Agent import Agent
from torch.utils.tensorboard import SummaryWriter
from Environment import Environment
from ReplayMemory import ReplayMemory


class Train:
    def __init__(self, utils):
        self.params = utils.params
        self.root_dir = self.params.STORE_DIR
        self.episode_num = int(self.params.EPISODE_NUM)
        self.batch_size = int(self.params.BATCH_SIZE)
        self.step_num = int(self.params.EPISODE_STEPS)
        self.device = self.params.DEVICE
        self.res_folder, self.res_name = utils.make_res_folder(root_dir=self.root_dir)
        self.log_dir = os.path.join(self.res_folder, 'log')
        self.tensor_writer = SummaryWriter()
        # self.tensorboard_call_back = CallBack(res_dir=self.res_folder, log_freq=self.params.PRINT_REWARD_FREQ, )

    def train_policy(self):
        print('start')
        environment = Environment(params=self.params, few_many_objects=['few', 'many'])
        agent = Agent(params=self.params)
        for episode in range(self.episode_num):
            state, _ = environment.reset()
            episode_reward = 0
            dqn_reward = 0
            episode_loss = 0
            for step in range(self.step_num):
                test_environment = deepcopy(environment)
                goal_map, dqn_goal_map = agent.get_action(state=state, episode=episode)

                new_state, reward, terminated, truncated, _ = environment.step(goal_map)
                _, test_reward, _, _, _ = test_environment.step(dqn_goal_map)

                # ('init_state', 'goal_map', 'reward', 'next_state')
                agent.save_experience(state, goal_map, reward, new_state)
                episode_reward += reward
                dqn_reward += test_reward
                episode_loss += agent.optimize()
                if episode % self.params.UPDATE_TARGET_NET:
                    agent.update_target_net()
                state = deepcopy(new_state)
            self.tensor_writer.add_scalar("Loss", episode_loss / self.step_num, episode)
            self.tensor_writer.add_scalar("Reward", episode_reward / self.step_num, episode)
            self.tensor_writer.add_scalar("DQN Reward", dqn_reward / self.step_num, episode)

        agent.save(self.res_folder)
