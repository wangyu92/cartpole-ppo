import torch
import torch.nn.functional as F
import numpy as np

from types import SimpleNamespace

def merge_exp(exp_queues):
    states, actions, rewards, dones, statesn = [[] for _ in range(5)]
    for queue in exp_queues:
        exp_dict = queue.get()
        exp_dict = SimpleNamespace(**exp_dict)

        states += exp_dict.states.tolist()
        statesn += exp_dict.statesn.tolist()
        actions += exp_dict.actions.tolist()
        rewards += exp_dict.rewards.tolist()
        dones += exp_dict.dones.tolist()

    states, actions, rewards, dones, statesn =\
        map(lambda x: np.array(x), [states, actions, rewards, dones, statesn])

    exp_dict = {
        'states': states,
        'actions': actions,
        'rewards': rewards,
        'dones': dones,
        'statesn': statesn,
    }

    return exp_dict


def merge_metric(metric_queues):

    episode_rewards = []

    for queue in metric_queues:
        metric_dict = queue.get()
        episode_rewards += metric_dict['episode_rewards']

    metric_dict = {
        'episode_rewards': episode_rewards
    }

    return metric_dict


def compute_gae(exp_dict, p, norm=True):
    exp = SimpleNamespace(**exp_dict)

    td_target = exp.rewards + p.GAMMA * exp.valuesn * (1 - exp.dones)
    delta = td_target - exp.values
    gae = np.append(np.zeros_like(exp.rewards), np.zeros((1, 1)), axis=0)
    for i in reversed(range(len(exp.rewards))):
        gae[i] = p.GAMMA * p.LAMBDA * gae[i + 1] * (1 - exp.dones[i]) + delta[i]
    gae = gae[:-1]

    gae = (gae - gae.mean()) / (gae.std() + 1e-8)

    return gae, td_target


def shuffle_exp(exp_dict):
    def shuffle_sample(x, seed):
        np.random.seed(seed)
        np.random.shuffle(x)

    exp = SimpleNamespace(**exp_dict)
    seed = np.random.randint(2**32 - 1)
    shuffle_sample(exp.states, seed)
    shuffle_sample(exp.actions, seed)
    shuffle_sample(exp.probs, seed)
    shuffle_sample(exp.rewards, seed)
    shuffle_sample(exp.dones, seed)
    shuffle_sample(exp.advs, seed)
    shuffle_sample(exp.returns, seed)

    return exp.__dict__


def split_exp(exp_dict, num_batch):
    exp = SimpleNamespace(**exp_dict)

    dicts = []
    exp.states = np.split(exp.states, num_batch)
    exp.actions = np.split(exp.actions, num_batch)
    exp.probs = np.split(exp.probs, num_batch)
    exp.rewards = np.split(exp.rewards, num_batch)
    exp.dones = np.split(exp.dones, num_batch)
    exp.values = np.split(exp.values, num_batch)
    exp.advs = np.split(exp.advs, num_batch)
    exp.returns = np.split(exp.returns, num_batch)

    for i in range(len(exp.states)):
        d = {
            'states': exp.states[i],
            'actions': exp.actions[i],
            'probs': exp.probs[i],
            'rewards': exp.rewards[i],
            'dones': exp.dones[i],
            'values': exp.values[i],
            'advs': exp.advs[i],
            'returns': exp.returns[i],
        }
        dicts.append(d)

    return dicts


def train_model(model, exp_dict, num_updates, p):
    exp_dicts = split_exp(exp_dict, p.NUM_MINIBATCH)

    losses = []
    losses_p = []
    losses_v = []
    ent_coefs = []
    entropies = []
    for _ in range(p.NUM_EPOCH):
        for i in range(p.NUM_MINIBATCH):
            results = train_step(model, exp_dicts[i], num_updates, p)
            losses.append(results[0])
            losses_p.append(results[1])
            losses_v.append(results[2])
            ent_coefs.append(results[3])
            entropies.append(results[4])

    res_dict = {
        'total': np.mean(losses),
        'actor': np.mean(losses_p),
        'critic': np.mean(losses_v),
        'ent_coefs': np.mean(ent_coefs),
        'entropies': np.mean(entropies)
    }

    return res_dict


def train_step(model, exp_dict, num_updates, p):
    device = model.getdevice()

    exp = SimpleNamespace(**exp_dict)
    states = torch.tensor(exp.states, device=device).float()
    actions = torch.tensor(exp.actions, device=device).long()
    oldprobs = torch.tensor(exp.probs, device=device).float()
    advs = torch.tensor(exp.advs, device=device).float()
    returns = torch.tensor(exp.returns, device=device).float()
    oldvpred = torch.tensor(exp.values, device=device).float()

    # back propagation
    optimizer = torch.optim.Adam(model.parameters(), lr=p.LR)
    optimizer.zero_grad()
    ppreds, vpreds = model(states)

    train_exp = {
        'states': states,
        'actions': actions,
        'oldprobs': oldprobs,
        'advs': advs,
        'returns': returns,
        'oldvpred': oldvpred,
        'ppreds': ppreds,
        'vpreds': vpreds
    }

    loss_policy = loss_policy_fn(train_exp, p)
    loss_value = loss_value_fn(train_exp, p)
    vf_coef = 1. # default value is 0.5. but 1. is better.

    entropy = torch.distributions.Categorical(probs=ppreds).entropy().mean()
    ent_decay = p.ENT_MAX - num_updates * (p.ENT_MAX - p.ENT_MIN) / p.ENT_STEP
    ent_coef = np.clip(ent_decay, p.ENT_MIN, p.ENT_MAX)

    loss = (loss_policy) - (ent_coef * entropy) + (vf_coef * loss_value)
    loss.backward()
    optimizer.step()

    loss = loss.detach().item()
    loss_policy = loss_policy.detach().item()
    loss_value = loss_value.detach().item()

    return loss, loss_policy, loss_value, ent_coef, entropy.item()


def loss_policy_fn(exp_dict, p):
    exp = SimpleNamespace(**exp_dict)

    probs = torch.gather(exp.ppreds, dim=-1, index=exp.actions)
    ratio = torch.exp(torch.log(probs) - torch.log(exp.oldprobs))
    kl = F.kl_div(exp.oldprobs, probs, reduction='none')

    # Rollback
    if p.PLOSS_TYPE == 'rollback':
        pgloss1 = ratio * exp.advs
        pgloss2 = torch.where(
            ratio <= 1 - p.CLIPRANGE,
            -p.RB_ALPHA * ratio + (1 + p.RB_ALPHA) * (1 - p.CLIPRANGE),
            torch.where(
                ratio >= 1 + p.CLIPRANGE,
                -p.RB_ALPHA * ratio + (1 + p.RB_ALPHA) * (1 - p.CLIPRANGE),
                ratio
            )
        ) * exp.advs

        loss = -torch.min(pgloss1, pgloss2)

    # Trust-Resion
    elif p.PLOSS_TYPE == 'trust-resion':
        pgloss1 = ratio * exp.advs
        pgloss2 = torch.where(
            kl >= p.TR_DELTA,
            torch.tensor(1., device=probs.device),
            ratio
        ) * exp.advs

        loss = -torch.min(pgloss1, pgloss2)

    # Trust-Resion with Rollback (Truly PPO)
    elif p.PLOSS_TYPE == 'truly':
        pgloss1 = ratio * exp.advs
        pgloss2 = torch.where(
            (kl >= p.TRULY_DELTA) & (ratio * exp.advs >= exp.advs),
            p.TRULY_ALPHA * kl,
            torch.tensor(p.TRULY_DELTA, device=probs.device)
        )
        loss = -(pgloss1 - pgloss2)

    # default
    else:
        surr1 = ratio * exp.advs
        surr2 = torch.clamp(ratio, 1 - p.CLIPRANGE, 1 + p.CLIPRANGE) * exp.advs
        loss = -torch.min(surr1, surr2)

    return loss.mean()
    

def loss_value_fn(exp_dict, p):
    exp = SimpleNamespace(**exp_dict)

    if p.VLOSS_TYPE == 'clip':
        vpredclipped = exp.oldvpred + torch.clamp(exp.vpreds - exp.oldvpred, -p.CLIPRANGE, p.CLIPRANGE)
        vf_losses1 = torch.nn.MSELoss()(exp.vpreds, exp.returns)
        vf_losses2 = torch.nn.MSELoss()(vpredclipped, exp.returns)
        loss = torch.max(vf_losses1, vf_losses2)
    else:
        loss = torch.nn.MSELoss()(exp.vpreds, exp.returns)

    return loss.mean()







