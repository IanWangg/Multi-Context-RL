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
import time

class DSRActor(BaseActor):
    def __init__(self, config):
        BaseActor.__init__(self, config)
        self.config = config
        self.start()

    def _transition(self):
        if self._state is None:
            self._state = self._task.reset()
        config = self.config
        with config.lock:
            # print('DEBUGGING : ', np.array(self._state).shape, type(config.state_normalizer))
            # print('After normalizing : ', type(config.state_normalizer(self._state)), config.state_normalizer(self._state).shape)
            # print(config.state_normalizer([1, 2, 3]).shape)
            _, psi, q_values = self._network(config.state_normalizer(self._state))
        q_values = to_np(q_values).flatten()
        if self._total_steps < config.exploration_steps \
                or np.random.rand() < config.random_action_prob():
            action = np.random.randint(0, len(q_values))
        else:
            action = np.argmax(q_values)
        next_state, reward, done, info = self._task.step([action])
        
        entry = [self._state[0], action, reward[0], next_state[0], int(done[0]), info]
        self._total_steps += 1
        self._state = next_state
        return entry


class DSRAgent(BaseAgent):
    def __init__(self, config):
        BaseAgent.__init__(self, config)
        self.config = config
        config.lock = mp.Lock()
        
        self.loss_q_vec = []
        self.loss_psi_vec = []
        self.loss_vec = []
        self.returns = []

        self.replay = config.replay_fn()
        self.actor = DSRActor(config)

        self.network = config.network_fn()
        self.network.share_memory()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.network.state_dict())
        self.optimizer = config.optimizer_fn(self.network.parameters())
        self.w_optimizer = config.optimizer_fn(self.network.psi2q.parameters()) # only on w's

        self.actor.set_network(self.network)

        self.total_steps = 0
        self.batch_indices = range_tensor(self.replay.batch_size)

        try:
            self.is_wb = config.is_wb
        except:
            self.is_wb = False
        
        if(self.is_wb):
            wandb.init(entity="psurya", project="sample-project")
            wandb.watch_called = False

    def close(self):
        close_obj(self.replay)
        close_obj(self.actor)

    def eval_step(self, state):
        self.config.state_normalizer.set_read_only()
        state = self.config.state_normalizer(state)
        _, _, q = self.network(state)
        action = to_np(q.argmax(-1))
        self.config.state_normalizer.unset_read_only()
        return action

    def step(self):
        tic =  time.perf_counter()
        config = self.config

        # Store transitions in the buffer
        transitions = self.actor.step()
        experiences = []
        # tic = time.perf_counter()
        for state, action, reward, next_state, done, info in transitions:

            # Reporting training progress on stdout
            # self.record_online_return(info)
            
            # Recording train returns in list
            for i, info_ in enumerate(info):
                ret = info_['episodic_return']
                if ret is not None:
                    self.returns.append([self.total_steps, ret])
                    if(self.is_wb):
                        wandb.log({"steps_ret": self.total_steps, "returns": ret})
                    
            self.total_steps += 1
            reward = config.reward_normalizer(reward)

            # Putting these new entries in experiences
            experiences.append([state, action, reward, next_state, done])

        toc = time.perf_counter()
        # print(f'time for collecting transitions : {toc - tic:0.4f} seconds')

        # Add experiences to replay
        self.replay.feed_batch(experiences)

        # Start updating network parameters after exploration_steps
        if self.total_steps > self.config.exploration_steps:
            tic =  time.perf_counter()
            # Sampling from replay buffer
            experiences = self.replay.sample()
            states, actions, rewards, next_states, terminals = experiences

            # Normalizing stat values
            states = self.config.state_normalizer(states)
            next_states = self.config.state_normalizer(next_states)

            # Computing targets
            '''
            LOOK AT HERE
            '''
            phi_next, psi_next, q_next = self.target_network(next_states)
            psi_next = psi_next.detach()
            q_next = q_next.detach()
            if self.config.double_q:
                best_actions = torch.argmax(self.network(next_states), dim=-1)
                q_next = q_next[self.batch_indices, best_actions]
            else:
                q_next = q_next.max(1) # predicts max q values (vector of b) and corresponding arguments
                a_star = q_next[1]
                q_next = q_next[0]
                psi_next = psi_next[self.batch_indices, a_star, :]

            terminals = tensor(terminals)
            rewards = tensor(rewards)

            # Estimate q target
            q_next = self.config.discount * q_next * (1 - terminals)
            q_next.add_(rewards)

            # Estimate psi target (obtained from target n/w) # CHECK: should we get target from network instead?
            # print(psi_next.shape)
            psi_next = self.config.discount * psi_next * (1 - terminals.unsqueeze(1).repeat(1, psi_next.shape[1]))
            # print(psi_next.shape)
            # print(self.target_network(next_states)[0].shape)
            '''
            check here
            '''
            temp = phi_next[0]
            psi_next.add_(temp.view(temp.size(0), psi_next.size(1))) # Adding the prediction for present state.
            # try:
            #     psi_next.add_(temp) # Adding the prediction for present state.
            # except:
            #     psi_next.add_(temp.view(temp.size(0), psi_next.size(1))) # Adding the prediction for present state.


            # Computing estimates
            actions = tensor(actions).long()
            '''
            LOOK AT HERE
            '''
            _, psi, q = self.network(states)
            q = q[self.batch_indices, actions]
            psi = psi[self.batch_indices, actions, :]

            # Estimating the loss
            loss_q = (q_next - q).pow(2).mul(0.5).mean()
            loss_psi = config.c * (psi_next - psi).pow(2).mul(0.5).mean()
            loss = loss_q + loss_psi
            
            # Storing loss estimates
            self.loss_vec.append(loss.item())
            # self.loss_q_vec.append(loss_q.item())
            # self.loss_psi_vec.append(loss_psi.item())
            # if(self.is_wb):
            #     wandb.log({"steps_loss": self.total_steps, "loss": loss.item(), "loss_psi": loss_psi.item(), "loss_q": loss_q.item()})
            
            if(not np.isfinite(loss.item())):
                print(' loss has diverged!')
                import pdb;pdb.set_trace()
                return 
            
            toc = time.perf_counter()
            # print(f'time for calculating loss : {toc - tic:0.4f} seconds')

            # tic = time.perf_counter()
            if(config.freeze == 1):
                # Update all based on loss_psi
                self.optimizer.zero_grad()
                loss_psi.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.optimizer.step()
                    
                # Update only w parameters based on loss_q
                self.w_optimizer.zero_grad()
                loss_q.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.w_optimizer.step()
            elif(config.freeze == 2):
                # Update only w parameters based on loss_q
                self.w_optimizer.zero_grad()
                loss_q.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.w_optimizer.step()
            else: # freeze = 0
                 # Update all params based on whole loss
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.network.parameters(), self.config.gradient_clip)
                with config.lock:
                    self.optimizer.step()

            toc = time.perf_counter()
            # print(f'time spend for updating weights {toc - tic:0.4f} seconds')

        if self.total_steps / self.config.sgd_update_frequency % \
                self.config.target_network_update_freq == 0:
            self.target_network.load_state_dict(self.network.state_dict())

        toc =  time.perf_counter()
        # print(f'time for one step: {toc - tic:0.4f} seconds')