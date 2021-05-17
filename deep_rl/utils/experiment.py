import numpy as np
import torch
from tqdm import trange, tqdm
import sys
sys.path.append('../..')

from deep_rl.network import *

learning_rate_dqn = 2e-3 #from 0.05 - 0.001
max_step_dqn = 1e5
linear_schedule_dqn = 6e4

def dqn_feature(config, **kwargs):
    # pre-defined config must be passed to the function
    assert config is not None

    generate_tag(kwargs)
    kwargs.setdefault('log_level', 0)
    config.merge(kwargs)

    config.task_fn = lambda: Task(config.game)
    config.eval_env = config.task_fn()

    config.optimizer_fn = lambda params: torch.optim.RMSprop(params, config.lr)
    config.network_fn = lambda: DQNCNN(config.action_dim, SRIdentityBody(config.state_dim), \
                                      hidden_units=(2000,))
#     config.network_fn = lambda: VanillaNet(config.action_dim, FCBody(config.state_dim, hidden_units=(16,)))
    config.replay_fn = lambda: AsyncReplay(memory_size=int(1e5), batch_size=10)

    config.random_action_prob = LinearSchedule(1.0, 0.1, config.linear_schedule)
    config.discount = 0.9
    config.target_network_update_freq = 200
    config.exploration_steps = 0
    # config.double_q = True
    config.double_q = False
    config.sgd_update_frequency = 4
    config.gradient_clip = 5
    config.eval_interval = int(5e3)
    config.max_steps = config.max_steps
    config.async_actor = False
    agent = DQNAgent(config)

    #run_steps function below
    config = agent.config
    agent_name = agent.__class__.__name__
    t0 = time.time()
    # agent.step()
    # plt.figure(figsize=(10,4))
    while True:
        # print(agent.actor._task.env.envs[0].goal)
        if config.save_interval and not agent.total_steps % config.save_interval:
            agent.save('data/%s-%s-%d' % (agent_name, config.tag, agent.total_steps))
        if config.log_interval and not agent.total_steps % config.log_interval:
            t0 = time.time()
        if config.eval_interval and not agent.total_steps % config.eval_interval:
            agent.eval_episodes()
            pass
        if config.max_steps and agent.total_steps >= config.max_steps:
            return agent
            break
        agent.step()
        # plt.title('step: {}'.format(agent.total_steps), fontsize=20)
        # plt.imshow(agent.actor._task.env.envs[0].render(), cmap='Blues', )
        agent.switch_task()
    return agent