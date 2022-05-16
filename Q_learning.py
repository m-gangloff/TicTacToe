import numpy as np
import time
from tic_env import OptimalPlayer
from tqdm.notebook import tqdm

def empty_pos(state_arr):
    """
    return all empty positions of a grid
    
    Args:
        state_arr: immutable array representing the current grid which is used as a key for the Q_table
    returns:
        List containing all available actions
    """
    avail = []
    for i in range(9):
        pos = (int(i/3), i % 3)
        if state_arr[i] == 0:
            avail.append(pos)
    return avail


def randomMove(grid):
    """ 
    Chose a random move from the available options.

    Args:
        grid: np.array of shape (3,3) describing the current game
    returns:
        Randomly chosen valid action
    """
    avail = empty_pos(grid)
    # np.random.randint: [start, end)
    return avail[np.random.randint(0, len(avail))]



def get_inner_dict(Q_table, state_arr, valid_actions):
    """
    Returns the inner dictionnary containing all the available actions for a state.
    If there is no corresponding entry for the state in the dictionnary, then a new entry is added

    Args:
        Q_table: dict() having as key the state as an array and as value a dict() with all the available actions
        state_arr: immutable array representing the current grid which is used as a key for the Q_table
        valid_actions: List of availabe valid actions for the current game state
    returns: 
        Inner dict() of the Q_table containing the available actions for the current state
    """
    try:
        inner_dict = Q_table[state_arr]
    except:
        inner_dict = {valid_action: 0 for valid_action in valid_actions}
        Q_table[state_arr] = inner_dict
    return inner_dict


def pick_action(state_arr, Q_table, epsilon=0):
    """
    Implements greedy policy and returns the action with max. Q-value (given the state).
    note: when Q-table is filled with zeros, returns a random policy. 

    Args:
        state_arr: immutable array representing the current grid which is used as a key for the Q_table
        Q_table: dict() having as key the state as an array and as value a dict() with all the available actions
        epsilon: Value between 0 and 1 which describes the probability to make a random move.
    returns:
        Depending on epsilon the random or best action to take or None if there is no available action. 
    """

    inner_dict = get_inner_dict(Q_table, state_arr, empty_pos(state_arr))

    if np.random.random() < epsilon:
        return randomMove(state_arr)

    if len(inner_dict) > 0:
        return max(inner_dict, key=lambda x : inner_dict[x])

    return None


def update_q(reward, Q_table, state_arr, action, next_state_arr=None, next_action=None, alpha=0.5, gamma=0.99):
    """
    Updates the Q_table using the iterative update rule

    Args:
        reward: Value representing the obtained reward
        Q_table: ``dict()`` having as key the state as an array and as value a ``dict()`` with all the available actions
        state_arr: immutable array representing the grid before the last step
        action: action taken at the last step
        state_arr: immutable array representing the grid after the last step, None if the game has finished
        next_action: action to take after at the last step, None if the game has finished
    """
    if state_arr==None or next_action==None:
        Q_table[state_arr][action] += alpha * (reward - Q_table[state_arr][action])
    else:
        Q_table[state_arr][action] += alpha * (reward + gamma * Q_table[next_state_arr][next_action] - Q_table[state_arr][action])


def init_game(env, players, eps_opt, eps_agent, Q_table):
    """
    Initialises the game, plays the first move if the optimal player begins and chooses the first action of the agent

    Args:
        env: Instance of the TictactoeEnv class from ``tic_env.py``
        players: List corresponding to the players of the game
        eps_opt: Probability of the optimal player to make a random move
        eps_agent: Probability of the agent to make a random move
        Q_table: ``dict()`` having as key the state as an array and as value a ``dict()`` with all the available actions
    returns:
        A tuple with the optimal plyer, the state of the game, the first action chosen by the agent 
        and a boolean corresponding to the end state of the game 
    """
    env.reset()
    # Change starting player after every game
    players.reverse()
    player_opt = OptimalPlayer(eps_opt, player=players[0])
    agent = players[1]
    state, end, _ = env.observe()
    if env.current_player == player_opt.player:
        action_opt = player_opt.act(state)
        state, end, _ = env.step(action_opt)
        
    state = tuple(state.flatten())
    action = pick_action(state, Q_table, epsilon=eps_agent)

    return player_opt, agent, state, action, end


def print_game_end(env, game_nb, winner, players):
    """
    Prints the game stats after it ended

    Args:
        env: Instance of the TictactoeEnv class from ``tic_env.py``
        game_nb: Current game number
        winner: Winner of the game, None if there is no winner
        players: List corresponding to the players of the game
    """
    print('-------------------------------------------')
    print('Game {} ended, winner is player {}'.format(game_nb, str(winner)))
    print('Optimal player = ' +  players[0])
    print('Agent player = ' +  players[1])
    env.render()


def performance_measures(env, Q_table):
    # Test vs. optimal policy
    _, rewards_vs_opt = eps_policy(env, Q_table=Q_table, nb_epochs=500, eps_agent=0, eps_opt=0, test_perf=True)
    M_opt = np.mean(rewards_vs_opt)
    # Test vs. random policy
    _, rewards_vs_rnd = eps_policy(env, Q_table=Q_table, nb_epochs=500, eps_agent=0, eps_opt=1, test_perf=True)
    M_rnd = np.mean(rewards_vs_rnd)
    return M_opt, M_rnd


def eps_policy(env, Q_table, nb_epochs=20000, eps_agent=0., eps_opt=0.5, alpha=0.5, gamma=0.99, decay_eps=False, 
                eps_min=0.1, eps_max=0.8, expl_games=1000, test_opt_vs_rnd=False, test_perf=False):
    """
    Trains an agent to play Tic Tac Toe using an epsilon-greedy policy. 
    See section 6.5 of https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

    Args:
    - env: Instance of the TictactoeEnv class from ```tic_env.py```
    - Q_table: Dict() representing the Q_table. Should be set to dict() when training a new agent.
    - nb_epochs: Number of games played
    - eps_agent: Probability of the agent to make a random move.
    - eps_opt: Probability of the optimal player to make a random move
    - alpha: Learning rate, value between 0 and 1
    - gamma: Discount factor, value between 0 and 1
    - decay_eps: If True, uses the exploration function ```eps(n) = max(eps_min, eps_max*(1 - n/n*)```
    - eps_min : minimum exploration value of the exploration function
    - eps_max : maximum exploration value of the exploration function
    - expl_games : number of games ```n*``` used for the denominator of the exploration function
    - test_opt_vs_rnd: If true, the performance between the agent and the optimal player will be tested after every 250 games
    - test_perf: If True, updates the Q-Table, else plays the games without updating the Q-Table
    Returns:
        Copy of the Q-table and a list containing the average of the rewards from every 250 games
    """
    #players[0] -> OptimalPlayer
    #players[1] -> Agent
    players = ['X','O']

    t_start = time.time()
    rewards = []
    rewards_250 = []
    M_opts = []
    M_rnds = []

    # for epoch in tqdm(range(nb_epochs), disable=test_perf, mininterval=1, miniters=1):
    # for epoch in tqdm(range(nb_epochs)):
    for epoch in range(nb_epochs):

        if decay_eps:
            eps_agent = max(eps_min, eps_max*(1-(epoch+1)/expl_games))

        player_opt, agent, state, action, end = init_game(env, players, eps_opt, eps_agent, Q_table)
        
        while not end:
            next_state, end, winner = env.step(action)
            
            # Agent wins
            if end:
                if not test_perf:
                    update_q(env.reward(agent), Q_table, state, action)
                break

            # Optimal player
            action_opt = player_opt.act(next_state)
            next_state, end, winner = env.step(action_opt)
            
            # Optimal player wins
            if end:
                if not test_perf:
                    update_q(env.reward(agent), Q_table, state, action)
                break
            
            # Agent
            #env.render()
            next_state = tuple(next_state.flatten())
            next_action = pick_action(next_state, Q_table, epsilon=eps_agent)

            # update Q-table using the iterative update rule 
            if not test_perf:
                update_q(env.reward(agent), Q_table, state, action, next_state, next_action)

            state = next_state
            action = next_action
        
        # After every 250 games, test performance or calculate average rewards
        rewards.append(env.reward(agent))
        if (epoch+1)%250 == 0:
            if test_opt_vs_rnd:
                M_opt, M_rnd = performance_measures(env, Q_table)
                M_opts.append(M_opt)
                M_rnds.append(M_rnd)
            else:
                rewards_250.append(np.mean(rewards))
                rewards = []

    t_end = time.time()
    if not test_perf:
        print('Learning finished after {:.2f}s\nPlayed a total of {} games'.format((t_end - t_start), nb_epochs))

    if test_opt_vs_rnd:
        return Q_table.copy(), M_opts, M_rnds

    return Q_table.copy(), rewards_250


def eps_policy_self_practice(env, Q_table, nb_epochs=20000, eps_agents=0., alpha=0.5, gamma=0.99,
            decay_eps=True, eps_min=0.1, eps_max=0.8, expl_games=1000):
    """
    Trains an agent to play Tic Tac Toe using an epsilon-greedy policy. 
    See section 6.5 of https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

    Args:
    - env: Instance of the TictactoeEnv class from ```tic_env.py```
    - Q_table: Dict() representing the Q_table. Should be set to dict() when training a new agent.
    - nb_epochs: Number of games played
    - eps_agents: Probability of the agents to make a random move.
    - alpha: Learning rate, value between 0 and 1
    - gamma: Discount factor, value between 0 and 1
    - decay_eps: If True, uses the exploration function ```eps(n) = max(eps_min, eps_max*(1 - n/n*)```
    - eps_min : minimum exploration value of the exploration function
    - eps_max : maximum exploration value of the exploration function
    - expl_games : number of games ```n*``` used for the denominator of the exploration function
    Returns:
        Copy of the Q-table and a list containing the average of the rewards from every 250 games
    """

    players = ['X','O']

    t_start = time.time()
    M_opts = []
    M_rnds = []


    for epoch in tqdm(range(nb_epochs)):

        if decay_eps:
            eps_agents = max(eps_min, eps_max*(1-(epoch+1)/expl_games))

        env.reset()

        states = {'X': None, 'O': None}
        actions = {'X': None, 'O': None}

        state, end, _ = env.observe()
        
        
        while not end:
            for player in players:
                next_state = tuple(state.flatten())
                next_action = pick_action(next_state, Q_table, epsilon=eps_agents)
                
                if actions[player] is not None:
                    update_q(env.reward(player), Q_table, states[player], actions[player], next_state, next_action)

                    # state, end, _ = env.observe()
                    # states[player] = tuple(state.flatten())
                    # actions[player] = pick_action(states[player], Q_table, epsilon=eps_agent)

                states[player] = next_state
                actions[player] = next_action
                
                state, end, winner = env.step(next_action)
                
                if end:
                    update_q(env.reward(players[0]), Q_table, states[players[0]], actions[players[0]])
                    update_q(env.reward(players[1]), Q_table, states[players[1]], actions[players[1]])
                    break
        
        # After every 250 games, test performance or calculate average rewards
        if (epoch+1)%250 == 0:
            M_opt, M_rnd = performance_measures(env, Q_table)
            M_opts.append(M_opt)
            M_rnds.append(M_rnd)

    t_end = time.time()
    print('Learning finished after {:.2f}s, played {} games'.format((t_end - t_start), nb_epochs))

    return Q_table.copy(), M_opts, M_rnds