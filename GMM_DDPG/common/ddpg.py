from common.value_networks import *
from common.policy_networks import *
import torch.optim as optim

import random


class DDPG():
    def __init__(self, args, replay_buffer, state_space, action_space):
        device_idx = args.device_idx
        if device_idx >= 0:
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device("cpu")
        print("Total device: ", self.device)

        self.replay_buffer = replay_buffer
        self.hidden_1 = args.hidden_1
        self.hidden_2 = args.hidden_2
        self.hidden_3 = args.hidden_3

        # single-branch network structure as in 'Memory-based control with recurrent neural networks'
        self.qnet = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.target_qnet = QNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.policy_net = DPG_PolicyNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)
        self.target_policy_net = DPG_PolicyNetwork(device_idx, state_space, action_space, self.hidden_1, self.hidden_2, self.hidden_3, args.n_layers, args.drop_prob).to(self.device)

        # two-branch network structure as in 'Sim-to-Real Transfer of Robotic Control with Dynamics Randomization'
        # self.qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_qnet = QNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)
        # self.target_policy_net = DPG_PolicyNetworkLSTM(state_space, action_space, hidden_dim).to(device)

        print('Q network: ', self.qnet)
        print('Policy network: ', self.policy_net)

        for target_param, param in zip(self.target_qnet.parameters(), self.qnet.parameters()):
            target_param.data.copy_(param.data)
        self.q_criterion = nn.MSELoss()
        q_lr=args.rate
        policy_lr= args.prate
        self.update_cnt= 0
        self.soft_tau = args.tau
        self.discount = args.discount
        self.q_optimizer = optim.Adam(self.qnet.parameters(), lr=q_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)
    
    def target_soft_update(self, net, target_net):
    # Soft update the target net
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(  # copy data value into target parameters
                target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau
            )

        return target_net

    def update(self, batch_size, gamma=0.95, target_update_delay=1, warmup=True):
        self.update_cnt+=1
        gamma = self.discount
        state, action, last_action, reward, next_state, done = self.replay_buffer.sample(batch_size)
        # print('sample:', state, action,  reward, done)
#        print(state[0][0][0], ",       ", len(state[1]))
        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        last_action     = torch.FloatTensor(last_action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(-1).to(self.device)  
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(-1).to(self.device)


        new_next_action = self.target_policy_net.evaluate(next_state)  # for q
        predict_target_q = self.target_qnet(next_state, new_next_action)  # for q
        target_q = reward+(1-done)*gamma*predict_target_q # for q



        # Critic update
        self.q_optimizer.zero_grad()

        predict_q = self.qnet(state, action) # for q 
        q_loss = self.q_criterion(predict_q, target_q.detach()) # + random.uniform(-1e-10, 1e-10)
#        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        if q_loss != q_loss:
            print("q_loss: ", q_loss)
            print("-----------------------\n", "predict_q: ", predict_q)
            print(predict_q != predict_q)
            print("state: ", state)
            print(state != state)
            print("action: ", action)
            print(action != action)
            print("=======================\n", "Target_q: ", target_q.detach())
            print(target_q.detach() != target_q.detach())
            print("reward: ", reward)
            print(reward != reward)
            print("predict_target_q: ", predict_target_q)
            print(predict_target_q != predict_target_q)
            print("next_state: ", next_state)
            print(next_state != next_state)
            print("new_next_action: ", new_next_action)
            print(new_next_action != new_next_action)
        q_loss.backward()
        self.q_optimizer.step()

        # Actor update
        self.policy_optimizer.zero_grad()
        new_action = self.policy_net.evaluate(state) # for policy
        predict_new_q = self.qnet(state, new_action) # for q 

#        self.q_optimizer.zero_grad()
        policy_loss = -torch.mean(predict_new_q)
#        self.policy_optimizer.zero_grad()

        # train qnet
#        q_loss.backward(retain_graph=True)  # no need for retain_graph here actually
        policy_loss.backward()

#        self.q_optimizer.step()

        # train policy_net     
        self.policy_optimizer.step()

        # update the target_qnet
        if self.update_cnt%target_update_delay==0:
            self.target_qnet=self.target_soft_update(self.qnet, self.target_qnet)
            self.target_policy_net=self.target_soft_update(self.policy_net, self.target_policy_net)

        return q_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

    def save_model(self, path):
        torch.save(self.qnet.state_dict(), '{}/qnet.pkl'.format(path) )
        torch.save(self.target_qnet.state_dict(), '{}/target_q.pkl'.format(path) )
        torch.save(self.policy_net.state_dict(), '{}/policy.pkl'.format(path) )

    def load_model(self, path):
        self.qnet.load_state_dict(torch.load('{}/qnet.pkl'.format(path), map_location=self.device))
        self.target_qnet.load_state_dict(torch.load('{}/target_q.pkl'.format(path), map_location=self.device))
        self.policy_net.load_state_dict(torch.load('{}/policy.pkl'.format(path), map_location=self.device))
#        self.qnet.eval()
#        self.target_qnet.eval()
#        self.policy_net.eval()
