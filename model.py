import math
import torch
from torch import nn
from torch.nn import functional as F


# NoisyLinear layer
class NoisyLinear(nn.Module):
  def __init__(self, in_features, out_features, std_init=0.4):
    super(NoisyLinear, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.std_init = std_init
    self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
    self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
    self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
    self.bias_mu = nn.Parameter(torch.empty(out_features))
    self.bias_sigma = nn.Parameter(torch.empty(out_features))
    self.register_buffer('bias_epsilon', torch.empty(out_features))
    self.reset_parameters()
    self.reset_noise()

  def reset_parameters(self):
    mu_range = 1 / math.sqrt(self.in_features)
    self.weight_mu.data.uniform_(-mu_range, mu_range)
    self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
    self.bias_mu.data.uniform_(-mu_range, mu_range)
    self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

  def _scale_noise(self, size):
    x = torch.randn(size)
    return x.sign().mul_(x.abs().sqrt_())

  def reset_noise(self):
    epsilon_in = self._scale_noise(self.in_features)
    epsilon_out = self._scale_noise(self.out_features)
    self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
    self.bias_epsilon.copy_(epsilon_out)

  def forward(self, input):
    if self.training:
      return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon, self.bias_mu + self.bias_sigma * self.bias_epsilon)
    else:
      return F.linear(input, self.weight_mu, self.bias_mu)


class DQN(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = x.view(-1, 3136)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q
  
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class DQN_rs(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space
    
    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    
    # region-sensitive module (iput_chan, output_chan, kernel_size)
    self.conv1_attent = nn.Conv2d(64, 512, 1)
    self.conv2_attent = nn.Conv2d(512, 2, 1)
    
    self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.normalize(x, p=2, dim=1)
    batch_size = x.size(0)
    weights = F.elu(self.conv1_attent(x))
    weights = self.conv2_attent(weights).view(-1, 2, 49)
    weights = F.softmax(weights.view(batch_size*2,-1), dim=1) #2D tensor by default is also dim 1
    weights = weights.view(batch_size,2,7,7)
    
    #Broadcasting
    x1 = x * weights[:, :1, :, :]
    x2 = x * weights[:, 1:, :, :]
    x = x1 + x2
    
    x = x.view(-1, 3136)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q # shape: (-1, self.action_space, self.atoms)
  
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

class DQN_rs_sig(nn.Module):
  def __init__(self, args, action_space):
    super().__init__()
    self.atoms = args.atoms
    self.action_space = action_space

    self.conv1 = nn.Conv2d(args.history_length, 32, 8, stride=4, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
    self.conv3 = nn.Conv2d(64, 64, 3)
    
    # region-sensitive module (iput_chan, output_chan, kernel_size)
    self.conv1_attent = nn.Conv2d(64, 512, 1)
    self.conv2_attent = nn.Conv2d(512, 2, 1)
    
    self.fc_h_v = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_h_a = NoisyLinear(3136, args.hidden_size, std_init=args.noisy_std)
    self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
    self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

  def forward(self, x, log=False):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = F.relu(self.conv3(x))
    x = F.normalize(x, p=2, dim=1)
    batch_size = x.size(0)
    weights = F.elu(self.conv1_attent(x))
    weights = self.conv2_attent(weights) # (batch, 2, 7, 7)
    weights = torch.sigmoid(weights)
    
    #Broadcasting
    x1 = x * weights[:, :1, :, :]
    x2 = x * weights[:, 1:, :, :]
    x = x1 + x2
    
    x = x.view(-1, 3136)
    v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
    a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream
    v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
    q = v + a - a.mean(1, keepdim=True)  # Combine streams
    if log:  # Use log softmax for numerical stability
      q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension
    else:
      q = F.softmax(q, dim=2)  # Probabilities with action over second dimension
    return q
  
  def reset_noise(self):
    for name, module in self.named_children():
      if 'fc' in name:
        module.reset_noise()

