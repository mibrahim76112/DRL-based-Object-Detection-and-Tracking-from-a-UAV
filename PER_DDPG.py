import numpy as np
import torch
import torch.nn.functional as F
from networks import Actor, Critic

torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class DDPG(object):
    def __init__(self, s_dim, a_dim, a_max):
        #Create actor and actor target 
        self.actor = Actor(s_dim, a_dim, a_max).to(device)
        self.actor_target = Actor(s_dim, a_dim, a_max).to(device)
        #Initialize actor and actor target exactly the same
        self.actor_target.load_state_dict(self.actor.state_dict())
        #Adam optimizer to train actor
        #Learning rate specified in DDPG paper
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.002)
        
        #Create critic and critic target
        self.critic = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim,trainable=False).to(device)
        #Initialize critic and critic target exactly the same
        self.critic_target.load_state_dict(self.critic.state_dict())
        #Adam optimizer to train critic
        #L2 weight decay specified in DDPG paper
        #print(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.004)
    
    #Given a state, the actor returns a policy 
    def get_action(self, s):
        s = torch.FloatTensor(s.reshape(1, -1)).to(device)
        return self.actor(s).cpu().data.numpy().flatten()

    #Update actor, critic and target networks with minibatch of experiences
    def train(self, replay_buffer, prioritized, beta_value, epsilon, batch_size=64, gamma=0.99, tau=0.005):
        if len(replay_buffer)<batch_size*3:
            return
        
        # Sample replay buffer
        if prioritized: 
            #Prioritized experience replay
            experience = replay_buffer.sample(batch_size, beta_value)
            s, a, r, s_new, done, weights, batch_idxes = experience
            #reshape data
            r = r.reshape(-1, 1)
            done = done.reshape(-1, 1)
            #We do not use importance sampling weights
            #Therefore importance sampling weights are all set to 1
            #See Hyperparameter search in report 
            weights = np.ones_like(r)
            #weights = weights.reshape(-1, 1)
        else:
            #Uniform experience replay
            s, a, r, s_new, done = replay_buffer.sample(batch_size)
            #importance sampling weights are all set to 1
            weights, batch_idxes = np.ones_like(r), None

        #Sqrt weights 
        #We do this since each weight will squared in MSE loss
        weights = np.sqrt(weights)

        #convert data to tensors
        state = torch.FloatTensor(s).to(device)
        action = torch.FloatTensor(a).to(device)
        next_state = torch.FloatTensor(s_new).to(device)
        done = torch.FloatTensor(1 - done).to(device)
        reward = torch.FloatTensor(r).to(device)
        weights = torch.FloatTensor(weights).to(device)

        #Compute the Q value estimate of the target network
        Q_target = self.critic_target(next_state, self.actor_target(next_state))
        #Compute Y
        Y = reward + (done * gamma * Q_target).detach()
        #Compute Q value estimate of critic
        Q = self.critic(state, action)
        #Calculate TD errors
        TD_errors = (Y - Q)
        #Weight TD errors 
        weighted_TD_errors = torch.mul(TD_errors, weights)
        #Create a zero tensor
        zero_tensor = torch.zeros(weighted_TD_errors.shape)
        #Compute critic loss, MSE of weighted TD_r
        critic_loss = F.mse_loss(weighted_TD_errors,zero_tensor)

        #Update critic by minimizing the loss
        #https://pytorch.org/docs/stable/optim.html
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute actor loss
        actor_loss = -self.critic(state, self.actor(state)).mean()
        #Update the actor policy using the sampled policy gradient:
        #https://pytorch.org/docs/stable/optim.html
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update the target models
        for critic_weights, critic__target_weights in zip(self.critic.parameters(), self.critic_target.parameters()):
            critic__target_weights.data.copy_(tau * critic_weights.data + (1 - tau) * critic__target_weights.data)
        for actor_weights, actor__target_weights in zip(self.actor.parameters(), self.actor_target.parameters()):
            actor__target_weights.data.copy_(tau * actor_weights.data + (1 - tau) * actor__target_weights.data)

        #For prioritized exprience replay
        #Update priorities of experiences with TD errors
        if prioritized:
            td_errors = TD_errors.detach().numpy()
            new_priorities = np.abs(td_errors) + epsilon
            replay_buffer.update_priorities(batch_idxes, new_priorities)
