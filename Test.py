import math

# from stable_baselines3 import PPO
import os
from sbx import PPO
import torch
import numpy as np
import itertools

from Agent import Agent
from Environment import Environment
import matplotlib.pyplot as plt


def get_predefined_parameters(params, param_name):
    if param_name == 'all_mental_states':
        all_param = [[-10, -5, 0, 5, 10]] * params.OBJECT_TYPE_NUM
    elif param_name == 'all_object_rewards':
        # all_param = [[0, 4, 8, 12, 16, 20]] * num_object
        param_range = params.ENVIRONMENT_OBJECT_REWARD_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1], num=min(param_range[1] - param_range[0] + 1, 4),
                                               dtype=int), axis=0).tolist() * params.OBJECT_TYPE_NUM
    elif param_name == 'all_mental_states_change':
        # all_param = [[0, 1, 2, 3, 4, 5]] * num_object
        param_range = params.MENTAL_STATES_SLOPE_RANGE
        all_param = np.expand_dims(np.linspace(param_range[0],
                                               param_range[1],
                                               num=min(param_range[1] - param_range[0] + 1, 4), dtype=int),
                                   axis=0).tolist() * params.OBJECT_TYPE_NUM
    else:
        print('no such parameters')
        return
    num_param = len(all_param[0]) ** params.OBJECT_TYPE_NUM
    param_batch = []
    for i, ns in enumerate(itertools.product(*all_param)):
        param_batch.append(list(ns))
    return param_batch


class Test:
    def __init__(self, utils):
        self.params = utils.params
        self.res_folder = utils.res_folder
        self.agent = self.load_model(self.params)
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.height = utils.params.HEIGHT
        self.width = utils.params.WIDTH
        self.object_type_num = utils.params.OBJECT_TYPE_NUM

        self.all_mental_states = get_predefined_parameters(self.params, 'all_mental_states')
        self.all_object_rewards = get_predefined_parameters(self.params, 'all_object_rewards')
        self.all_mental_states_change = get_predefined_parameters(self.params, 'all_mental_states_change')

        self.color_options = [[1, 0, .2], [0, .8, .2], [0, 0, 0]]
        self.goal_shape_options = ['*', 's', 'P', 'o', 'D', 'X']
        self.objects_color_name = ['red', 'green', 'black']  # 2: stay

    def get_figure_title(self, mental_states):
        title = '$n_{0}: {1:.2f}'.format('{' + self.objects_color_name[0] + '}', mental_states[0])
        for i in range(1, self.object_type_num):
            title += ", n_{0}: {1:.2f}$".format('{' + self.objects_color_name[i] + '}', mental_states[i])
        return title

    def get_object_shape_dictionary(self, object_locations, agent_location, each_type_object_num):
        shape_map = dict()
        for obj_type in range(self.object_type_num):
            at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
            for at_obj in range(each_type_object_num[obj_type]):
                key = tuple(at_type_object_locations[at_obj, 1:].tolist())
                shape_map[key] = self.goal_shape_options[at_obj]
        key = tuple(agent_location)
        shape_map[key] = '.'
        return shape_map


    def get_goal_location_from_goal_map(self, goal_map):
        goal_location = np.argwhere(goal_map)[0]
        return goal_location

    def next_agent_and_environment(self):
        for object_reward in self.all_object_rewards:
            for mental_state_slope in self.all_mental_states_change:
                environment = Environment(self.params, ['few', 'many'])

                for subplot_id, mental_state in enumerate(self.all_mental_states):
                    for i in range(self.height):
                        for j in range(self.width):
                            env, state, object_locations, each_type_object_num = environment.init_environment_for_test(
                                [i, j],
                                mental_state,
                                mental_state_slope,
                                object_reward)
                            env_parameters = [mental_state, mental_state_slope, object_reward]
                            yield env, state, [i,
                                                  j], object_locations, each_type_object_num, env_parameters, subplot_id

    def get_goal_directed_actions(self):
        fig, ax = None, None
        row_num = 5
        col_num = 5
        for setting_id, outputs in enumerate(self.next_agent_and_environment()):
            environment = outputs[0]
            state = outputs[1]
            agent_location = outputs[2]
            object_locations = outputs[3]
            each_type_object_num = outputs[4]
            env_parameters = outputs[5]  # [mental_state, mental_state_slope, object_reward]
            subplot_id = outputs[6]

            if setting_id % (col_num * row_num * self.width * self.height) == 0:
                fig, ax = plt.subplots(row_num, col_num, figsize=(15, 12))

            r = subplot_id // col_num
            c = subplot_id % col_num

            ax[r, c].set_xticks([])
            ax[r, c].set_yticks([])
            ax[r, c].invert_yaxis()

            shape_map = self.get_object_shape_dictionary(object_locations, agent_location, each_type_object_num)

            with torch.no_grad():
                goal_map = self.agent.get_action(state=state, episode=0, epsilon=-1)
                goal_location = self.get_goal_location_from_goal_map(goal_map)

            if tuple(goal_location.tolist()) in shape_map.keys():
                selected_goal_shape = shape_map[tuple(goal_location.tolist())]
                goal_type = np.where(environment[:, goal_location[0], goal_location[1]])[0].min()
            else:
                selected_goal_shape = '_'
                goal_type = 0

            goal_type = 2 if goal_type == 0 else goal_type - 1
            size = 10 if selected_goal_shape == '.' else 50
            ax[r, c].scatter(agent_location[1], agent_location[0],
                             marker=selected_goal_shape,
                             s=size,
                             alpha=0.4,
                             facecolor=self.color_options[goal_type])

            if agent_location[0] == self.height - 1 and agent_location[1] == self.width - 1:
                ax[r, c].set_title(self.get_figure_title(env_parameters[0]), fontsize=10)

                for obj_type in range(self.object_type_num):
                    at_type_object_locations = object_locations[object_locations[:, 0] == obj_type]
                    for obj in range(each_type_object_num[obj_type]):
                        ax[r, c].scatter(at_type_object_locations[obj, 1:][1],
                                         at_type_object_locations[obj, 1:][0],
                                         marker=self.goal_shape_options[obj],
                                         s=200,
                                         edgecolor=self.color_options[obj_type],
                                         facecolor='none')
                ax[r, c].tick_params(length=0)
                ax[r, c].set(adjustable='box')
            if (setting_id + 1) % (col_num * row_num * self.width * self.height) == 0:
                plt.tight_layout(pad=0.1, w_pad=6, h_pad=1)
                fig.savefig('{0}/slope_{1}-{2}_or_{3}-{4}.png'.format(self.res_folder,
                                                                      env_parameters[1][0],
                                                                      env_parameters[1][1],
                                                                      env_parameters[2][0],
                                                                      env_parameters[2][1]))
                plt.close()

    def load_model(self, params):
        model_path = os.path.join(self.res_folder, 'model.pt')
        model_parameters = torch.load(model_path)
        agent = Agent(params)
        agent.policy_net.load_state_dict(model_parameters)
        return agent

    # @staticmethod
    # def get_qfunction_selected_goal_map(state,
    #                                     agent: Agent):
    #
    #     goal_map, goal_location = agent.get_action(state=state, episode=0, epsilon=-1)  # get the goal map based on Q-values
    #     new_state, reward, terminated, truncated, _ = environment.step(goal_map)
    #
    #     return torch.tensor(rho).mean()