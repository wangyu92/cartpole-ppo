import os
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import numpy as np
import gym
import datetime
import copy

import params
import train_func

from types import SimpleNamespace
from torch.utils.tensorboard import SummaryWriter

from model import Model

TB_SCALAR       = 'scalar'
TB_HIST         = 'histogram'


def tensorboard(tb_queue):
    stime = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    dir_path = '../tensorboard/' + 'CartPole-PPO_' + stime
    writer = SummaryWriter(dir_path)

    writer.add_text('hparams', str(params.data), 0)

    while True:
        tbtype, tag, value, step = tb_queue.get()
        if tbtype == TB_SCALAR:
            writer.add_scalar(tag, value, step)
        elif tbtype == TB_HIST:
            writer.add_histogram(tag, value, step)


def test(test_queue, tb_queue):
    device = torch.device('cuda')

    # --- Create Model ---
    env = gym.make('CartPole-v1')

    while True:
        model = test_queue.get()
        model.to(device)
        model.eval()

        p = SimpleNamespace(**model.params)

        episode_rewards = [0.]

        for _ in range(p.TEST_ITER):
            s = env.reset()
            while True:
                a = model.action(s)
                s, r, d, _ = env.step(a)
                episode_rewards[-1] += r
                if d:
                    episode_rewards.append(0.)
                    break

        tb_queue.put((TB_SCALAR, 'Test/Reward', np.mean(episode_rewards), p.NUM_UPDATE))



def central(exp_queues, model_queues, metric_queues, test_queue, tb_queue):
    device = torch.device('cuda')
    p = params.get()

    # --- Create Model ---
    env = gym.make('CartPole-v1')
    sdim = env.observation_space.shape
    adim = env.action_space.n
    model = Model(sdim[0], adim)

    while True:
        # --- Synchronize model: send the model to each agent ---
        model.to(torch.device('cpu'))
        state_dict = model.state_dict()
        for i in range(p.NUM_AGENTS):
            model_queues[i].put(state_dict)
        model.to(device)
        model.train()

        # --- Receive, compute, shuffle exp ---
        exp = train_func.merge_exp(exp_queues)
        exp['values'] = model.values(exp['states'])
        exp['valuesn'] = model.values(exp['statesn'])
        exp['probs'] = model.probs(exp['states'], exp['actions'])
        exp['advs'], exp['returns'] = train_func.compute_gae(exp, p=p, norm=True)
        exp = train_func.shuffle_exp(exp)

        # --- Update model ---
        results = train_func.train_model(model, exp, p.NUM_UPDATE, p=p)
        r = SimpleNamespace(**results)
        p.NUM_UPDATE += 1

        # --- Send to tensorboard ---
        tbs = []
        tbs.append((TB_SCALAR, 'Loss(updates)/Total', r.total, p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Actor', r.actor, p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Critic', r.critic, p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Entropy Coef', r.ent_coefs, p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Entropy', r.entropies, p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Returns', np.mean(exp['returns']), p.NUM_UPDATE))
        tbs.append((TB_SCALAR, 'Loss(updates)/Advantage', np.mean(exp['advs']), p.NUM_UPDATE))
        for d in tbs:
            tb_queue.put(d)

        metric_dict = train_func.merge_metric(metric_queues)
        metric = SimpleNamespace(**metric_dict)
        for i in range(len(metric.episode_rewards)):
            tb_queue.put((TB_SCALAR, 'Episode/Rewards', metric.episode_rewards[i], p.NUM_EPISODE))
            p.NUM_EPISODE += 1

        # --- update model param ---
        model.params = p.__dict__

        # --- test model ---
        if test_queue.empty():
            model.to(torch.device('cpu'))
            test_queue.put(copy.deepcopy(model))


def agent(i, exp_queue, model_queue, metric_queue):
    device = torch.device('cuda')
    p = params.get()

    # --- Create environments ---
    env = gym.make('CartPole-v1')
    sdim = env.observation_space.shape
    adim = env.action_space.n

    # --- Create model ---
    model = Model(sdim[0], adim)

    # --- exp data ---
    states, statesn = [np.empty((p.HORIZON,) + sdim) for _ in range(2)]
    rewards, dones = [np.empty((p.HORIZON, 1)) for _ in range(2)]
    actions = np.empty((p.HORIZON, 1), dtype=np.int)

    episode_rewards = [0.]

    sn = env.reset()
    while True:
        # --- update model ---
        state_dict = model_queue.get()
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        # --- get exp ---
        for t in range(p.HORIZON):
            states[t] = sn.copy()
            actions[t] = model.action_sample(states[t])
            sn, rewards[t], dones[t], _ = env.step(actions[t][0])
            episode_rewards[-1] += rewards[t]

            if dones[t]:
                sn = env.reset()
                episode_rewards.append(0.)

            statesn[t] = sn.copy()

        # --- send exp ---
        exp_dict = {
            'states': states,
            'statesn': statesn,
            'rewards': rewards,
            'actions': actions,
            'dones': dones,
        }
        exp_queue.put(exp_dict)

        metric_dict = {
            'episode_rewards': episode_rewards,
        }
        metric_queue.put(metric_dict)


if __name__ == '__main__':
    p = params.get()

    ###########################################################################
    # Create queues for communication                                         #
    ###########################################################################
    exp_queues = []
    model_queues = []
    metric_queues = []
    for i in range(p.NUM_AGENTS):
        exp_queues.append(mp.Queue(1))
        model_queues.append(mp.Queue(1))
        metric_queues.append(mp.Queue(1))

    test_queue = mp.Queue(1)
    tb_queue = mp.Queue(1024)

    ###########################################################################
    # Create processes                                                        #
    ###########################################################################
    args = (exp_queues, model_queues, metric_queues, test_queue, tb_queue)
    central = mp.Process(target=central, args=args)
    central.start()

    agents = []
    for i in range(p.NUM_AGENTS):
        args = (i, exp_queues[i], model_queues[i], metric_queues[i])
        agents.append(mp.Process(target=agent, args=args))

    for i in range(p.NUM_AGENTS):
        agents[i].start()

    args = (test_queue, tb_queue)
    test_process = mp.Process(target=test, args=args)
    test_process.start()

    args = (tb_queue,)
    tb_process = mp.Process(target=tensorboard, args=args)
    tb_process.start()

    ###########################################################################
    # Join processes                                                          #
    ###########################################################################
    central.join()

    for i in range(p.NUM_AGENTS):
        agents[i].join()

    test_process.join()
    tb_process.join()
