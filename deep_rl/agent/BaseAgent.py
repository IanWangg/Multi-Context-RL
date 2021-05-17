#######################################################################
# Copyright (C) 2017 Shangtong Zhang(zhangshangtong.cpp@gmail.com)    #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import torch
import numpy as np
from ..utils import *
import torch.multiprocessing as mp
from collections import deque
from skimage.io import imsave
import matplotlib.pyplot as plt

from deep_rl.component.fourrooms import FourRoomsMatrix

class BaseAgent:
    def __init__(self, config):
        self.config = config
        self.logger = get_logger(tag=config.tag, log_level=config.log_level)
        self.task_ind = 0

    def close(self):
        close_obj(self.task)

    def save(self, filename):
        torch.save(self.network.state_dict(), '%s.model' % (filename))
        with open('%s.stats' % (filename), 'wb') as f:
            pickle.dump(self.config.state_normalizer.state_dict(), f)

    def load(self, filename):
        state_dict = torch.load('%s.model' % filename, map_location=lambda storage, loc: storage)
        self.network.load_state_dict(state_dict)
        with open('%s.stats' % (filename), 'rb') as f:
            self.config.state_normalizer.load_state_dict(pickle.load(f))

    def eval_step(self, state):
        raise NotImplementedError

    def eval_episode(self):
        env = self.config.eval_env
        state = env.reset()
        while True:
            # print('still evaluating')
            action = self.eval_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            if ret is not None:
                # print('done')
                break
        return ret

    def eval_episodes(self):
        episodic_returns = []
        for ep in range(self.config.eval_episodes):
            total_rewards = self.eval_episode()
            episodic_returns.append(np.sum(total_rewards))
        self.logger.info('steps %d, episodic_return_test %.2f(%.2f)' % (
            self.total_steps, np.mean(episodic_returns), np.std(episodic_returns) / np.sqrt(len(episodic_returns))
        ))
        self.logger.add_scalar('episodic_return_test', np.mean(episodic_returns), self.total_steps)
        return {
            'episodic_return_test': np.mean(episodic_returns),
        }

    def record_online_return(self, info, offset=0):
        if isinstance(info, dict):
            ret = info['episodic_return']
            if ret is not None:
                self.logger.add_scalar('episodic_return_train', ret, self.total_steps + offset)
                self.logger.info('steps %d, episodic_return_train %s' % (self.total_steps + offset, ret))
        elif isinstance(info, tuple):
            for i, info_ in enumerate(info):
                self.record_online_return(info_, i)
        else:
            raise NotImplementedError

    def switch_task(self):
        config = self.config
        if not config.tasks:
            return
        print('switching task')
        segs = np.linspace(0, config.max_steps, len(config.tasks) + 1)
        if self.total_steps > segs[self.task_ind + 1]:
            self.task_ind += 1
            self.task = config.tasks[self.task_ind]
            self.states = self.task.reset()
            self.states = config.state_normalizer(self.states)

    def record_episode(self, dir, env):
        mkdir(dir)
        steps = 0
        state = env.reset()
        while True:
            self.record_obs(env, dir, steps)
            action = self.record_step(state)
            state, reward, done, info = env.step(action)
            ret = info[0]['episodic_return']
            steps += 1
            if ret is not None:
                break

    def record_step(self, state):
        raise NotImplementedError

    # For DMControl
    def record_obs(self, env, dir, steps):
        env = env.env.envs[0]
        obs = env.render(mode='rgb_array')
        imsave('%s/%04d.png' % (dir, steps), obs)

    def check(self, init=None, frame=20):
        self.actor.check(init)

    def mapping(self):
        self.actor.mapping()


class BaseActor(mp.Process):
    STEP = 0
    RESET = 1
    EXIT = 2
    SPECS = 3
    NETWORK = 4
    CACHE = 5

    def __init__(self, config):
        mp.Process.__init__(self)
        self.config = config
        self.__pipe, self.__worker_pipe = mp.Pipe()

        self._state = None
        self._task = None
        self._network = None
        self._total_steps = 0
        self.__cache_len = 2

        if not config.async_actor:
            self.start = lambda: None
            self.step = self._sample
            self.close = lambda: None
            self._set_up()
            self._task = config.task_fn()

    def _sample(self):
        transitions = []
        for _ in range(self.config.sgd_update_frequency):
            transitions.append(self._transition())
        return transitions

    def run(self):
        self._set_up()
        config = self.config
        self._task = config.task_fn()

        cache = deque([], maxlen=2)
        while True:
            op, data = self.__worker_pipe.recv()
            if op == self.STEP:
                if not len(cache):
                    cache.append(self._sample())
                    cache.append(self._sample())
                self.__worker_pipe.send(cache.popleft())
                cache.append(self._sample())
            elif op == self.EXIT:
                self.__worker_pipe.close()
                return
            elif op == self.NETWORK:
                self._network = data
            else:
                raise NotImplementedError

    def _transition(self):
        raise NotImplementedError

    def check(self, init=None, frame = 25):
        goal = self._task.env.envs[0].goal
        env = FourRoomsMatrix(goal=goal)
        # state = env.reset(init = init)
        # self._state = list(state)
        # plt.figure()
        # plt.subplot(1, 3, 1)
        # self._state = env.reset(init=init)
        # print(env.reset(init = init), env.config)
        # plt.imshow(env.render().reshape([13, 13]))
        # action = self._transition()[1]
        # plt.subplot(1, 3, 2)
        # env.step(action)
        # plt.imshow(env.render().reshape([13, 13]))
        # plt.subplot(1, 3, 3)
        # self._state = env.render_state().flatten()
        # action = self._transition()[1]
        # env.step(action)
        # plt.imshow(env.render().reshape([13, 13]))
        fig, axs = plt.subplots(5, 5, dpi=200)
        fig.suptitle(f'Behaviour Checking For {self.__class__.__name__} with goal = {goal}')
        for i in range(25):
            if i == 0:
                state = env.reset(init = init)
            else:
                action = self._transition()[1]
                env.step(action)
            self._state = env.render_state().flatten()
            x = i % 5
            y = i // 5
            axs[y, x].imshow(env.render().reshape([13, 13]))
    
    def mapping(self):
        goal = self._task.env.envs[0].goal
        env = FourRoomsMatrix(goal=goal)
        plt.figure()
        plt.suptitle(f'Behaviour Visualization For {self.__class__.__name__} wiht goal = {goal}')
        
        # Four possible actions
        # 0: UP
        # 1: DOWN
        # 2: LEFT
        # 3: RIGHT
        direction = [np.array((1,0)), np.array((-1,0)), np.array((0,-1)), np.array((0,1))]
        _map = (env.occupancy).flatten()
        # print(_map)
        occupancy = env.occupancy.flatten()
        state_index = 0
        for i in range(len(_map)):
            if occupancy[i] == 1:
                _map[i] = -1
                continue
            # now the state we are looking at is an available state
            if goal == state_index:
                _map[i] = 10
            env.reset(init = state_index)
            self._state = [env.reset(init = state_index)]
            action = self._transition()[1]
            _map[i] = action # check (the number represent the color)
            if goal == state_index:
                _map[i] = 10
            state_index += 1
            
        
        plt.imshow(_map.reshape([13, 13]))
        state_index = 0
        U = np.zeros(169)
        V = np.zeros(169)
        for i in range(len(_map)):
            if occupancy[i] == 1:
                _map[i] = -1
                continue
            # now the state we are looking at is an available state
            env.reset(init = state_index)
            self._state = [env.reset(init = state_index)]
            state_index += 1
            action = self._transition()[1]
            V[i] = direction[action][0]
            U[i] = direction[action][1]
    
        plt.colorbar()
        X = [i for i in range(13)]
        Y = [i for i in range(13)]
        U = U.reshape([13, 13])
        V = V.reshape([13, 13])
        plt.quiver(X, Y, U, V)

        print(np.array(_map).reshape([13, 13]))
        # quiver([X, Y], U, V, [C], **kw)
        # print(np.array(_map).reshape([13, 13]))
        # print(U)
        # print(V)
            

    def _set_up(self):
        pass

    def step(self):
        self.__pipe.send([self.STEP, None])
        return self.__pipe.recv()

    def close(self):
        self.__pipe.send([self.EXIT, None])
        self.__pipe.close()

    def set_network(self, net):
        if not self.config.async_actor:
            self._network = net
        else:
            self.__pipe.send([self.NETWORK, net])
