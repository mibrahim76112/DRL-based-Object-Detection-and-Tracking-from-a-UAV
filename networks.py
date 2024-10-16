import torch
import torch.nn as nn
import torch.nn.functional as F
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Architecture from experimental details section of the DDPG 
#paper "Continuous control with deep reinforcement learning"
#Lillicrap et. al. 2015
class Critic_Wasay(nn.Module):
    def __init__(self, s_dim, a_dim):
        super(Critic_Wasay, self).__init__()
        self.l1 = nn.Linear(s_dim+a_dim, 400)
        self.l2 = nn.Linear(400, 300)
        self.l3 = nn.Linear(300, 1)
        self.device=device
    def forward(self, s, a):
        x = torch.cat([s, a],1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, a_max):
        super(Actor, self).__init__()
        self.a_max = a_max
        self.device=device
        
        self.l1 = nn.Linear(s_dim, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, a_dim)
  
    def forward(self, s):
        x = F.relu(self.l1(s))
        x = F.relu(self.l2(x))
        x = torch.tanh(self.l3(x)) * self.a_max  
        return x 

#Architecture from experimental details section of the DDPG
#paper "Continuous control with deep reinforcement learning"
#Lillicrap et. al. 2015
class Critic(nn.Module):
    def __init__(self, s_dim, a_dim,trainable=True):
        super(Critic, self).__init__()
        self.device=device
        self.W_s=nn.Parameter(torch.randn(s_dim,64),requires_grad=trainable)
        self.W_a=nn.Parameter(torch.randn(a_dim,64),requires_grad=trainable)
        self.b1=nn.Parameter(torch.randn(1,64),requires_grad=trainable)
        self.l1 = nn.Linear(64, 64)
        self.l2 = nn.Linear(64, 1)
        


    def forward(self, s, a):
        x = F.relu(torch.matmul(s,self.W_s)+torch.matmul(a,self.W_a)+self.b1)
        x = F.relu(self.l1(x))
        return self.l2(x)
