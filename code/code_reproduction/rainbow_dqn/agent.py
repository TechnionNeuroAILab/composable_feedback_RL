# Port of Kaixhin/Rainbow agent.py. learn() is the original n-step distributional Double-Q update with
# prioritized-replay importance weights, generalised to `branches` independent action branches (branches=1 is
# numerically identical to the original, see test_port.py). PORT changes are marked.
import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_

from model import DQN


class Agent():
  def __init__(self, args, action_space, branches=1, obs_dim=None):
    self.action_space = action_space
    self.branches = branches  # PORT
    self.atoms = args.atoms
    self.Vmin = args.V_min
    self.Vmax = args.V_max
    self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
    self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
    self.batch_size = args.batch_size
    self.n = args.multi_step
    self.discount = args.discount
    self.norm_clip = args.norm_clip

    self.online_net = DQN(args, self.action_space, branches, obs_dim).to(device=args.device)
    self.online_net.train()

    self.target_net = DQN(args, self.action_space, branches, obs_dim).to(device=args.device)
    self.update_target_net()
    self.target_net.train()
    for param in self.target_net.parameters():
      param.requires_grad = False

    self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

  # Resets noisy weights in all linear layers (of online net only)
  def reset_noise(self):
    self.online_net.reset_noise()

  # Acts based on single state (no batch). Returns one action per branch (a python int when branches == 1)
  def act(self, state):
    with torch.no_grad():
      a = (self.online_net(state.unsqueeze(0)) * self.support).sum(3).argmax(2)[0]  # (branches,)
      return a.item() if self.branches == 1 else a.cpu().numpy()

  # Acts with an epsilon-greedy policy (used for evaluation only)
  def act_e_greedy(self, state, epsilon=0.001):  # High epsilon can reduce evaluation scores drastically
    if np.random.random() < epsilon:
      return np.random.randint(0, self.action_space) if self.branches == 1 else np.random.randint(0, self.action_space, self.branches)
    return self.act(state)

  def learn(self, mem):
    # Sample transitions
    idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
    B, nb, atoms = self.batch_size, self.branches, self.atoms
    actions = actions.view(B, nb)

    # Calculate current state probabilities (online network noise already sampled)
    log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ., ; theta_online): (B, nb, A, atoms)
    log_ps_a = log_ps.gather(2, actions.view(B, nb, 1, 1).expand(B, nb, 1, atoms)).squeeze(2)  # log p(s_t, a_t): (B, nb, atoms)

    with torch.no_grad():
      # Calculate nth next state probabilities
      pns = self.online_net(next_states)  # Probabilities p(s_t+n, ., ; theta_online)
      dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, .; theta_online))
      argmax_indices_ns = dns.sum(3).argmax(2)  # Double-Q: argmax_a[(z, p(s_t+n, a; theta_online))]: (B, nb)
      self.target_net.reset_noise()  # Sample new target net noise
      pns = self.target_net(next_states)  # Probabilities p(s_t+n, ., ; theta_target)
      pns_a = pns.gather(2, argmax_indices_ns.view(B, nb, 1, 1).expand(B, nb, 1, atoms)).squeeze(2)  # (B, nb, atoms)
      pns_a = pns_a.reshape(B * nb, atoms)

      # Compute Tz (Bellman operator T applied to z); the n-step return and terminal mask are shared by all branches
      Tz = returns.view(B, 1, 1) + nonterminals.view(B, 1, 1) * (self.discount ** self.n) * self.support.view(1, 1, atoms)
      Tz = Tz.expand(B, nb, atoms).reshape(B * nb, atoms)
      Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
      # Compute L2 projection of Tz onto fixed support z
      b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / delta_z
      l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
      # Fix disappearing probability mass when l = b = u (b is int)
      l[(u > 0) * (l == u)] -= 1
      u[(l < (atoms - 1)) * (l == u)] += 1

      # Distribute probability of Tz
      m = pns_a.new_zeros(B * nb, atoms)
      offset = (torch.arange(B * nb, device=m.device) * atoms).unsqueeze(1).expand(B * nb, atoms)
      m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
      m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)
      m = m.view(B, nb, atoms)

    loss = -torch.sum(m * log_ps_a, 2).mean(1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t))), mean over branches
    self.online_net.zero_grad()
    (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
    clip_grad_norm_(self.online_net.parameters(), self.norm_clip)  # Clip gradients by L2 norm
    self.optimiser.step()

    mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

  def update_target_net(self):
    self.target_net.load_state_dict(self.online_net.state_dict())

  # Evaluates Q-value based on single state (no batch)
  def evaluate_q(self, state):
    with torch.no_grad():
      return (self.online_net(state.unsqueeze(0)) * self.support).sum(3).max(2)[0].mean().item()

  def train(self):
    self.online_net.train()

  def eval(self):
    self.online_net.eval()
