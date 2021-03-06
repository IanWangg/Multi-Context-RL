#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

"""
DQN code modified to convert it to DSR by Surya.
The representations are not learned in this network.
"""

from ..network import *
from ..component import *
from ..utils import *
import time
from .BaseAgent import *
import wandb
import copy


class GPIActor(BaseActor):
    def __init__(self, config, agents):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()
        self.existed_agents = agents

    def _transition(self):
        # print(self._total_steps)
        if self._state is None:
            self._state = self._task.reset()
        # config = self.config
        # with config.lock:
        #     _, psi, q_values = self._network(config.state_normalizer(self._state))
        # q_values = to_np(q_values).flatten()
        # if self._total_steps < config.exploration_steps \
        #         or np.random.rand() < config.random_action_prob():
        #     action = np.random.randint(0, len(q_values))
        # else:
        #     action = np.argmax(q_values)
        # next_state, reward, done, info = self._task.step([action])
        #
        # entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        # self._total_steps += 1
        # self._state = next_state
        # return entry
        # print(self._state.shape)
        final_q_values = [float('-inf'), float('-inf'), float('-inf'), float('-inf')]
        q_values = []
        config = self.config
        with config.lock:
            for _agent in self.existed_agents:
                _, _,q_values = _agent.actor._network(config.state_normalizer(self._state))
                q_values = to_np(q_values).flatten()
                # print(q_values)
                final_q_values = np.maximum(q_values, final_q_values)
        # print(final_q_values)
        if np.random.rand() == config.random_action_prob():
            action = np.random.randint(0, len(final_q_values))
        else:
            action = np.argmax(final_q_values)
            
            
        # print(self.existed_agents[0].actor._network)
        # print(self.existed_agents[0].actor._network.psi2q.w)
        # print(action)
        next_state, reward, done, info = self._task.step([action])

        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]

        self._total_steps += 1
        self._state = next_state
        return entry




'''
This agent works totally depending on the ref agents provided. It does not has its own network
'''
class GPIAgent(BaseAgent):
    def __init__(self, config, agents, target):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()

        self.loss_q_vec = []
        self.loss_psi_vec = []
        self.loss_vec = []
        self.returns = []

        self.replay = config.replay_fn()
        self.agents = agents
        for _agent in self.agents:
            _agent.network.psi2q = copy.deepcopy(target.network.psi2q) # need an elegent way to do it
            _agent.actor._network.psi2q = copy.deepcopy(target.network.psi2q)
            # _agent.network.psi2q = type(target.network.psi2q)()
            # _agent.actor._network.psi2q = type(target.network.psi2q)()
            # _agent.network.psi2q.load_state_dict(target.network.psi2q.state_dict())
            # _agent.actor._network.psi2q.load_state_dict(target.network.psi2q.state_dict())
            # for i in range(len(_agent.network.layers)):
            #     print(i)
            #     _agent.network.layers[i].load_state_dict(target.network.layers[i].state_dict())
            #     _agent.actor._network.layers[i].load_state_dict(target.network.layers[i].state_dict())
        self.actor = GPIActor(config, self.agents)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.w_optimizer = config.optimizer_fn(self.network.psi2q.parameters())  # only on w's

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)



        try:
            self.is_wb = config.is_wb
        except:
            self.is_wb = False

        if (self.is_wb):
            wandb.init(entity="psurya", project="sample-project")
            wandb.watch_called = False



    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        
        # _, _, q = self.network(state)
        # action1 = to_np(q.argmax(-1))
        
        final_q_values = [float('-inf'), float('-inf'), float('-inf'), float('-inf')]
        q_values = []
        config = self.config
        with config.lock:
            for _agent in self.agents:
                _, _,q_values = _agent.actor._network(config.state_normalizer(state))
                q_values = to_np(q_values).flatten()
                # print(q_values)
                final_q_values = np.maximum(q_values, final_q_values)
        # print(final_q_values)
        
        action = to_np(torch.Tensor(final_q_values).reshape([1,4]).argmax(-1))
        
        if np.random.rand() == self.config.random_action_prob():
            action = np.array([np.random.randint(0, len(final_q_values))])
        else:
            action = to_np(torch.Tensor(final_q_values).reshape([1,4]).argmax(-1))
            
        
        # print(action, action1)
        # print(q.shape, torch.Tensor(final_q_values).shape)
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        config = self.config

        # Store transitions in the buffer
        transitions = self.actor.step()
        for _ in transitions:
            self.total_steps += 1
        # experiences = []
        # for state, action, reward, next_state, done, info in transitions:

        #     # Reporting training progress on stdout
        #     # self.record_online_return(info)

        #     # Recording train returns in list
        #     for i, info_ in enumerate(info):
        #         ret = info_['episodic_return']
        #         if ret is not None:
        #             self.returns.append([self.total_steps, ret])
        #             if (self.is_wb):
        #                 wandb.log({"steps_ret": self.total_steps, "returns": ret})

        #     self.total_steps += 1
        #     reward = config.reward_normalizer(reward)

        #     # Putting these new entries in experiences
        #     experiences.append([state, action, reward, next_state, done])

        # # Add experiences to replay
        # self.replay.feed_batch(experiences)

        # # Start updating network parameters after exploration_steps
        # if self.total_steps > self.config.exploration_steps:

        #     # Sampling from replay buffer
        #     experiences = self.replay.sample()
        #     states, actions, rewards, next_states, terminals = experiences

        #     # Normalizing stat values
        #     states = self.config.state_normalizer(states)
        #     next_states = self.config.state_normalizer(next_states)

        #     # Computing targets
        #     _, psi_next, q_next = self.target_network(next_states)
        #     psi_next = psi_next.detach()
        #     q_next = q_next.detach()
        #     if self.config.double_q:
        #         best_actions = torch.argmax(self.network(next_states), dim=-1)
        #         q_next = q_next[self.batch_indices, best_actions]
        #     else:
        #         q_next = q_next.max(1)  # predicts max q values (vector of b) and corresponding arguments
        #         a_star = q_next[1]
        #         q_next = q_next[0]
        #         psi_next = psi_next[self.batch_indices, a_star, :]

        #     terminals = tensor(terminals)
        #     rewards = tensor(rewards)

        #     # Estimate q target
        #     q_next = self.config.discount * q_next * (1 - terminals)
        #     q_next.add_(rewards)

        #     # Estimate psi target (obtained from target n/w) # CHECK: should we get target from network instead?
        #     psi_next = self.config.discount * psi_next * (1 - terminals.unsqueeze(1).repeat(1, psi_next.shape[1]))
        #     psi_next.add_(self.target_network(next_states)[0])  # Adding the prediction for present state.

        #     # Computing estimates
        #     actions = tensor(actions).long()
        #     _, psi, q = self.network(states)
        #     q = q[self.batch_indices, actions]
        #     psi = psi[self.batch_indices, actions, :]

        #     # Estimating the loss
        #     loss_q = (q_next - q).pow(2).mul(0.5).mean()
        #     loss_psi = config.c * (psi_next - psi).pow(2).mul(0.5).mean()
        #     loss = loss_q + loss_psi

        #     # Storing loss estimates
        #     self.loss_vec.append(loss.item())
        #     self.loss_q_vec.append(loss_q.item())
        #     self.loss_psi_vec.append(loss_psi.item())
        #     if (self.is_wb):
        #         wandb.log({"steps_loss": self.total_steps, "loss": loss.item(), "loss_psi": loss_psi.item(),
        #                   "loss_q": loss_q.item()})

        #     if (not np.isfinite(loss.item())):
        #         print(' loss has diverged!')
        #         import pdb;
        #         pdb.set_trace()
        #         return

        #     if (config.freeze == 1):
        #         # Update all based on loss_psi
        #         self.optimizer.zero_grad()
        #         loss_psi.backward(retain_graph=True)
        #         nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        #         with config.lock:
        #             self.optimizer.step()

        #         # Update only w parameters based on loss_q
        #         self.w_optimizer.zero_grad()
        #         loss_q.backward()
        #         nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        #         with config.lock:
        #             self.w_optimizer.step()
        #     elif (config.freeze == 2):
        #         # Update only w parameters based on loss_q
        #         self.w_optimizer.zero_grad()
        #         loss_q.backward()
        #         nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        #         with config.lock:
        #             self.w_optimizer.step()
        #     else:  # freeze = 0
        #         # Update all params based on whole loss
        #         self.optimizer.zero_grad()
        #         loss.backward(retain_graph=True)
        #         nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
        #         with config.lock:
        #             self.optimizer.step()

        # if self.total_steps / self.config.sgd_update_frequency % \
        #         self.config.target_network_update_freq == 0:
        #     self.target_network.load_state_dict(self.network.state_dict())