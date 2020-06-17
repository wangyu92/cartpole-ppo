import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Model(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Model, self).__init__()
        # init instance variables
        self.num_states = num_states
        self.num_actions = num_actions
        
        num_units = 128
        self.p_h1 = nn.Linear(self.num_states, num_units)
        self.p_h2 = nn.Linear(self.num_states, num_units)
        self.p_outs = nn.Linear(num_units, num_actions)

        self.v_h1 = nn.Linear(self.num_states, num_units)
        self.v_h2 = nn.Linear(self.num_states, num_units)
        self.v_outs = nn.Linear(num_units, 1)

    def forward(self, x):
        p = F.relu(self.p_h1(x))
        p = F.relu(self.p_h2(x))
        p = F.softmax(self.p_outs(p), dim=-1)

        v = F.relu(self.v_h1(x))
        v = F.relu(self.v_h2(x))
        v = self.v_outs(v)

        return p, v

    def action_sample(self, s):
        device = self.getdevice()

        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()
        s = torch.unsqueeze(s, axis=0)

        prob, _ = self(s)
        prob = prob.detach().cpu()

        dist = torch.distributions.Categorical(probs=prob)
        action = dist.sample().numpy()[0]
        return action

    def action(self, s):
        device = self.getdevice()

        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()
        s = torch.unsqueeze(s, axis=0)
        
        prob, _ = self(s)
        prob = prob.detach().cpu()

        action = torch.argmax(prob).detach().numpy()
        return action

    def values(self, s):
        device = self.getdevice()

        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()

        _, values = self(s)
        return values.detach().cpu().numpy()

    def value(self, s):
        device = self.getdevice()

        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()
        s = torch.unsqueeze(s, axis=0)

        _, values = self(s)
        return values.detach().cpu().numpy()[0, 0]

    def prob(self, s, a):
        device = self.getdevice()

        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()
        s = torch.unsqueeze(s, axis=0)

        if type(a).__module__ == np.__name__:
            a = torch.tensor(a.copy(), device=device).long()

        prob, _ = self(s)
        return prob[0, a].item()

    def probs(self, s, a):
        device = self.getdevice()
        if type(s).__module__ == np.__name__:
            s = torch.tensor(s.copy(), device=device).float()

        if type(a).__module__ == np.__name__:
            a = torch.tensor(a.copy(), device=device).long()

        preds, _ = self(s)
        probs = torch.gather(preds, dim=-1, index=a)

        return probs.detach().cpu().numpy()

    def getdevice(self):
        return torch.device(next(self.parameters()).device if next(self.parameters()).is_cuda else 'cpu')

if __name__ == '__main__':
    states = torch.zeros((2, 4))

    # predict
    model = ActorCritic(4, 2)
    print(model(states))

    # action_sample()
    state = np.zeros((4,))
    print(model.action_sample(state))

    # action()
    state = np.zeros((4,))
    print(model.action(state))