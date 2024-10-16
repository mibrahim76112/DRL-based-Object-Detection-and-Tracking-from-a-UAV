import torch as T
import torch.nn.functional as F
from networks import Actor, Critic
from buffer import ReplayBuffer

device = T.device("cuda" if T.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, alpha, beta, input_dims, tau, a_max,
                gamma=0.99,n_actions=2, max_size=1000000, batch_size=100):
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.actor = Actor(s_dim=input_dims, a_dim=n_actions,a_max=a_max)
        self.critic = Critic(s_dim=input_dims,a_dim=n_actions)

        self.target_actor = Actor(s_dim=input_dims, a_dim=n_actions,a_max=a_max)

        self.target_critic = Critic(s_dim=input_dims, a_dim=n_actions)

        self.actor.optimizer = T.optim.Adam(self.actor.parameters(), lr=alpha)
        self.critic.optimizer = T.optim.Adam(self.critic.parameters(),
                lr=beta)
        self.target_actor.optimizer = T.optim.Adam(self.target_actor.parameters(),  
                lr=alpha)
        self.target_critic.optimizer = T.optim.Adam(self.target_critic.parameters(),
                lr=beta)
        self.update_network_parameters(tau=1)

    def choose_action(self, s):
        self.actor.eval()
        s = T.FloatTensor(s.reshape(1, -1)).to(device)
        a=self.actor(s).cpu().data.numpy().flatten()
        self.actor.train()
        return a

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        if self.memory.mem_cntr < self.batch_size*3:
            return

        states, actions, rewards, states_, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        states_ = T.tensor(states_, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done,dtype=T.int).to(self.actor.device)

        target_actions = self.target_actor.forward(states_)
        critic_value_ = self.target_critic.forward(states_, target_actions)
        critic_value = self.critic.forward(states, actions)

        critic_value_[done] = 0.0
        critic_value_ = critic_value_.view(-1)

        target = rewards + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value)
        critic_loss.backward()
        self.critic.optimizer.step()

        self.actor.optimizer.zero_grad()
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
             actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)




