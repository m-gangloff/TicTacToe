import torch
import torch.nn as nn
import torch.optim as optim

from collections import namedtuple, deque
import numpy as np
from tqdm import tqdm 
import random
import time

from tic_env import TictactoeEnv, OptimalPlayer
from Q_learning import print_game

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """
    The queue saving all quadruplets ('state', 'action', 'next_state', 'reward') in the past
    """
    def __init__(self, capacity):
        """
        params: capacity: the maximum size of the queue
        """
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """ Save a transition """
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """ Sample transitions """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    """
    Deep Q-Learning as a MLP
    The inputs are encoded states, outputs are the Q-values for each possible actions
    """

    def __init__(self, state_size=18, hidden_size=128, n_hidden_layers=2, n_actions=9):
        """
        params:
            state_size: size of encoded states, act as input size
            hidden_size: size of hidden layers
            n_hidden_layers: number of hidden layers 
            n_actions: number of possible actions, act as output size
        """

        super(DQN, self).__init__()
        self.n_actions = n_actions
        
        layers = [nn.Linear(state_size, hidden_size)]

        for _ in range(n_hidden_layers-1):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_size,hidden_size))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_size, n_actions))

        self.layers = nn.Sequential(*layers)
        
    def forward(self, x):
        """
        params: x: encoded state tensor with shape (bacth_size, state_size)
        """
        return self.layers(x)

def update_eps(current_epoch, max_epoch=20000, eps_min=0.1, eps_max=0.8):
    return max(eps_min, eps_max*(1-current_epoch/max_epoch)) 

def pick_action(state, policy_net, epsilon=0, device='cuda'):
    """
    Implements epsilon-greedy policy and returns the action with max. Q-value.

    params:
        state: the encoded state with the expected shape (1, state_size)
        policy_net: the DQN module, inputs are encoded states, outputs are Q-values
        epsilon: Value between 0 and 1 which describes the probability to make a random move.
        device: 'cuda' or 'cpu', indicate the enviroment pytorch uses

    returns:
        Depending on epsilon the random or best action to take 
    """
    if np.random.random() < epsilon:
        return torch.tensor([[random.randrange(policy_net.n_actions)]], device=device, dtype=torch.long)

    with torch.no_grad():
        return policy_net(state.to(device)).max(1)[1].view(1, 1)

def grid_to_state_tensor(grid, our_player, device):
    """
    Transform the grid of env() (whose shape is (3,3)) 
    into encoded state tensor (whose shape is (1,18))

    params:
        grid: the numpy array represent current enviroment, 1 for X, -1 for O, otherwise O
        our_player: 'X' or 'O'

    returns:
        The flattent encoded state

    """

    gridX, gridO = grid.copy(), grid.copy() 

    # The value is 1 if X takes place, otherwise 0
    gridX[gridX<0]=0
    
    # The value is -1 if O takes place, otherwise 0
    gridO[gridO>0]=0
    # The value is 1 if O takes place, otherwise 0
    gridO=np.abs(gridO)
    
    # The first slice is our posistion, the second is the position of opponent
    if our_player=='X':
        state = torch.FloatTensor(np.stack([gridX, gridO],-1).reshape((1,-1))).to(device)
    else:
        state = torch.FloatTensor(np.stack([gridO, gridX],-1).reshape((1,-1))).to(device)
        
    return state


def setup_env(n_hidden=2, hidden_size=128, buffer_size=10000, lr=5e-4):
    """
    Initialises the enviroment,
    the deep Q-learning models (act as a Q-table), 
    the optimizer for them, 
    the data structure to store quadruplets ('state', 'action', 'next_state', 'reward')

    params :
        n_hidden: int, number of hidden layers of policy net 
        hidden_size: int, the size of hidden layers in policy net
        buffer_size: int, the maximum size of ReplayMemory
        lr: float, the learning rate of optimizer

    returns:
        enviroment: Instance of the TictactoeEnv class from ``tic_env.py``
        policy_net: the DQN to compute the Q-values,
        target_net: the DQN sharing weights with policy_net but freezed 
        optimizer: RMSoptimizer for policy_net
        memory: ReplayMemory to store history data 
        device: device that torch uses
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TictactoeEnv()
    policy_net = DQN(hidden_size=hidden_size, n_hidden_layers=n_hidden).to(device)
    target_net = DQN(hidden_size=hidden_size, n_hidden_layers=n_hidden).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.RMSprop(policy_net.parameters(), lr=lr)
    memory = ReplayMemory(buffer_size)
    
    return env, policy_net, target_net, optimizer, memory, device


def optimize_model(policy_net, target_net, optimizer, memory, batch_size=64, gamma=0.99, device='cuda'):
    """
    Optimize the policy_net by off-policy Qlearning

    params :
        policy_net: the policy net , used to compute Q(s,a)
        target_net: the target net , used to compute Q(s',a')
        optimizer: optimizer of policy net
        memory: ReplayMemory storing history data 
        batch_size:
        gamma: discount factor
        device: device that torch uses

    returns:
        The loss value

    """

    if len(memory) < batch_size:
        return
    
    transitions = memory.sample(batch_size)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))
    
    # Shape (batch_size, 18), (batch_size, 1), (batch_size,)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # Shape  (batch_size, 1)
    state_action_values = policy_net(state_batch.to(device)).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state batch_size final.
    # Shape  (batch_size, )
    next_state_values = torch.zeros(batch_size, device=device)

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    # Shape (batch_size,)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)

    # There're some non final states
    if torch.any(non_final_mask):

        # Shape (#non_final_next_states, 18)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])

        # For final states, the values are zeros
        next_state_values[non_final_mask] = target_net(non_final_next_states.to(device)).max(1)[0].detach()
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    loss_val = loss.item()
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
        
    optimizer.step()
    
    return loss_val


def train(eps_agent=0., eps_opt=0.5, gamma=0.99, alpha=5e-4,
        nb_epochs=20000, target_update=500, buffer_size=10000, 
        batch_size=64, n_hidden=2, hidden_size=128, 
        decay_eps=False, eps_min=0.1, eps_max=0.8, max_epoch=20000,
        eval_every=250, print_last_10_games=False, progress_bar=True):

    """
    Trains an agent to play Tic Tac Toe using an Deep Q Learning. 
    See more at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    params:
        eps_agent: Probability of the agent to make a random move
        eps_opt: Probability of the optimal player to make a random move
        alpha: Learning rate, value between 0 and 1
        gamma: Discount factor, value between 0 and 1
        nb_epochs: Number of games played
        target_update: Number of epoch between two consecutive times we update the weight of target net
        buffer_size: The maximum size of ReplayMemory
        batch_size: batch size for training the policy net
        n_hidden: Number of hidden layers of policy net 
        hidden_size: The size of hidden layers in policy net
        decay_eps: whether to decay the agent epsilon
        eps_min, eps_max, max_epoch: params of decay eps 

    returns:
        average training losses, average rewards, Mrand, Mopt for every 250 epochs
    """

    env, policy_net, target_net, optimizer, memory, device = setup_env(n_hidden, hidden_size, buffer_size, alpha)
    
    #players[0] -> OptimalPlayer
    #players[1] -> Agent
    players = ['X','O']

    t_start = time.time()

    avg_training_losses, training_losses = [], []
    avg_rewards, rewards = [], []
    Mrands, Mopts = [], []

    for epoch in tqdm(range(nb_epochs), disable=not progress_bar):
        
        if decay_eps:
            eps_agent = update_eps(epoch, max_epoch, eps_min, eps_max)

        # Reset enviroment before starting game
        env.reset()
        grid, end, _ = env.observe()

        # Switch the first players and init the optimal player
        players = np.flip(players)
        player_opt = OptimalPlayer(epsilon=eps_opt, player=players[0])

        # If the optimal player goes first
        if env.current_player == player_opt.player:
            move = player_opt.act(grid)
            grid, end, winner = env.step(move)

        # Encode the state
        state = grid_to_state_tensor(grid, players[1], device)

        while not end:
            # Select and perform an action
            action = pick_action(state, policy_net, eps_agent, device)

            # If agent takes an unavailable action
            if not env.check_valid(action.item()):
                # end the game and reward is -1 
                end = True
                reward = torch.tensor([-1], device=device)

            else:
                # Agent plays
                grid, end, _ = env.step(action.item())

                # If agent does not win yet
                if not end:
                    # Optimal player plays
                    move = player_opt.act(grid)
                    grid, end, winner = env.step(move) 
                # Get the reward after one round  
                reward = torch.tensor([env.reward(player=players[1])], device=device)

            if end:
                next_state = None
            else:
                next_state = grid_to_state_tensor(grid, players[1], device)
                
           # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the policy network)
            loss = optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device)
            
            if loss is not None:
                training_losses.append(loss)
            rewards.append(reward.item())

            if end:
                break
        
        if eval_every >0 and epoch % eval_every == eval_every-1:
            avg_training_losses.append(sum(training_losses)/len(training_losses))
            avg_rewards.append(sum(rewards)/len(rewards))
            training_losses, rewards = [], []

            Mopt, Mrand = evaluate(env, policy_net, device)
            Mrands.append(Mrand)
            Mopts.append(Mopt)

        if epoch % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if print_last_10_games and epoch>=nb_epochs-10:
            progress_bar = False
            print_game(env, epoch, winner, players)
    
    t_end = time.time()
    print('Learning finished after {:.2f}s\nPlayed a total of {} games'.format((t_end - t_start), nb_epochs))

    return policy_net, avg_training_losses, avg_rewards, Mrands, Mopts

def self_train(eps_agent=0., gamma=0.99, alpha=5e-4,
        nb_epochs=20000, target_update=500, buffer_size=10000, 
        batch_size=64, n_hidden=2, hidden_size=128, 
        decay_eps=False, eps_min=0.1, eps_max=0.8, max_epoch=20000,
        eval_every=250, print_last_10_games=False, progress_bar=True):

    """
    Trains an agent to play Tic Tac Toe using an Deep Q Learning. 
    See more at https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

    params:
        eps_agent: Probability of the agent to make a random move
        alpha: Learning rate, value between 0 and 1
        gamma: Discount factor, value between 0 and 1
        nb_epochs: Number of games played
        target_update: Number of epoch between two consecutive times we update the weight of target net
        buffer_size: The maximum size of ReplayMemory
        batch_size: batch size for training the policy net
        n_hidden: Number of hidden layers of policy net 
        hidden_size: The size of hidden layers in policy net
        decay_eps: whether to decay the agent epsilon
        eps_min, eps_max, max_epoch: params of decay eps 

    returns:
        average training losses, average rewards, Mrand, Mopt for every 250 epochs
    """

    env, policy_net, target_net, optimizer, memory, device = setup_env(n_hidden, hidden_size, buffer_size, alpha)
    
    players = ['X','O']

    t_start = time.time()

    avg_training_losses, training_losses = [], []
    avg_rewards, rewards = [], []
    Mrands, Mopts = [], []

    for epoch in tqdm(range(nb_epochs), disable=not progress_bar):
        
        if decay_eps:
            eps_agent = update_eps(epoch, max_epoch, eps_min, eps_max)

        # Reset enviroment before starting game
        env.reset()
        grid, end, _ = env.observe()
        
        # 0 for X and 1 for O
        states = {0: None, 1: None}
        next_states = {0: None, 1: None}
        actions = {0: None, 1: None}


        # Encode the current state X
        states[0] = grid_to_state_tensor(grid, 'X', device)

        while not end:
            for player_id, player in enumerate(players):

                # Select and perform an action X
                actions[player_id] = pick_action(states[player_id], policy_net, eps_agent, device)

                # If agent takes an unavailable action
                if not env.check_valid(actions[player_id].item()):
                    # end the game and reward is -1 
                    end = True
                    reward = torch.tensor([-1], device=device)
                    memory.push(states[player_id], actions[player_id], None, reward)
                    rewards.append(reward.item()) 

                else:
                    # Agent X plays
                    grid, end, _ = env.step(actions[player_id].item()) 
                    
                    # grid is intermediate state of X, but is the next state of O
                    if end:
                        next_states[1-player_id] = None 
                    else:
                        next_states[1-player_id] = grid_to_state_tensor(grid, players[1-player_id], device)

                    # If agent O plays before, prepare data to push into memory
                    if states[1-player_id] is not None and actions[1-player_id] is not None:
                        reward = torch.tensor([env.reward(player=players[1-player_id])], device=device)
                        memory.push(states[1-player_id], actions[1-player_id], next_states[1-player_id], reward)
                        rewards.append(reward.item())

                    # Update the state of O
                    states[1-player_id] = next_states[1-player_id]

                    if end:
                        reward = torch.tensor([env.reward(player=players[player_id])], device=device)
                        memory.push(states[player_id], actions[player_id], None, reward)
                        rewards.append(reward.item())
                        
                # Perform one step of the optimization (on the policy network)
                loss = optimize_model(policy_net, target_net, optimizer, memory, batch_size, gamma, device)
                
                if loss is not None:
                    training_losses.append(loss)
                
                if end:
                    break
            
        if eval_every >0 and epoch % eval_every == eval_every-1:
            avg_training_losses.append(sum(training_losses)/len(training_losses))
            avg_rewards.append(sum(rewards)/len(rewards))
            training_losses, rewards = [], []

            Mopt, Mrand = evaluate(env, policy_net, device)
            Mrands.append(Mrand)
            Mopts.append(Mopt)

        if epoch % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if print_last_10_games and epoch>=nb_epochs-10:
            progress_bar = False
            print_game(env, epoch, winner, players)
    
    t_end = time.time()
    print('Learning finished after {:.2f}s\nPlayed a total of {} games'.format((t_end - t_start), nb_epochs))

    return policy_net, avg_training_losses, avg_rewards, Mrands, Mopts

def evaluate(env, policy_net, device):
    """
    Compute Mrand and Mopt

    params:
        env: Instance of the TictactoeEnv class from ``tic_env.py``
        policy_net: the DQN module, inputs are encoded states, outputs are Q-values
        device: 'cuda' or 'cpu', indicate the device pytorch uses

    returns:
        Mopt, Mrand
        
    """
    #players[0] -> OptimalPlayer
    #players[1] -> Agent
    players = ['X','O']

    results = []
    eps_agent = 0
    for eps_opt in range(2):
        n_wins, n_losses = 0, 0
        for _ in range(500):
            # Reset enviroment before starting game
            env.reset()
            grid, end, _ = env.observe()

            # Switch the first players and init the optimal player
            players = np.flip(players)
            player_opt = OptimalPlayer(epsilon=eps_opt, player=players[0])

            # If the optimal player goes first
            if env.current_player == player_opt.player:
                move = player_opt.act(grid)
                grid, end, winner = env.step(move)

            # Encode the state
            state = grid_to_state_tensor(grid, players[1], device)

            while not end:
                # Select and perform an action
                action = pick_action(state, policy_net, eps_agent, device)

                # If agent takes an unavailable action
                if not env.check_valid(action.item()):
                    # end the game and reward is -1 
                    end = True
                    n_losses += 1
                    break

                # Agent plays
                grid, end, _ = env.step(action.item())
                
                # Agent wins
                if end:
                    n_wins +=1
                    break 
                
                # If agent does not win yet
                # Optimal player plays
                move = player_opt.act(grid)
                grid, end, winner = env.step(move) 
                
                # Optimal player wins
                if end:
                    n_losses +=1
                    break 

                next_state = grid_to_state_tensor(grid, players[1], device)
    
                # Move to the next state
                state = next_state
   
        results.append((n_wins-n_losses)/500)

    return results
