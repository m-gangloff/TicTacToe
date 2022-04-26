import numpy as np
import time
from tic_env import OptimalPlayer


def empty_pos(state_arr):
    """
    return all empty positions of a grid
    
    params:
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

    params:
        grid: np.array of shape (3,3) describing the current game
    returns:
        Randomly chosen valid action
    """
    avail = empty_pos(grid)
    return avail[np.random.randint(0, len(avail)-1)]


def get_inner_dict(Q_table, state_arr, valid_actions):
    """
    Returns the inner dictionnary containing all the available actions for a state.
    If there is no corresponding entry for the state in the dictionnary, then a new entry is added

    params:
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

    params:
        state_arr: immutable array representing the current grid which is used as a key for the Q_table
        Q_table: dict() having as key the state as an array and as value a dict() with all the available actions
        epsilon: Value between 0 and 1 which describes the probability to make a random move.
    returns:
        Depending on epsilon the random or best action to take or None if there is no available action. 
    """
    if np.random.random() < epsilon:
        return randomMove(state_arr)

    # Get all the valid actions converted to integers between 0 and 8
    valid_actions = empty_pos(state_arr)

    # Create a tuple with the corresponding actions and Q values
    inner_dict = get_inner_dict(Q_table, state_arr, valid_actions)

    if len(inner_dict) > 0:
        return max(inner_dict, key=lambda x : inner_dict[x])

    return None


def update_q(reward, Q_table, state_arr, action, next_state_arr=None, next_action=None, alpha=0.5, gamma=0.99):
    """
    Updates the Q_table using the iterative update rule

    params:
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

    params:
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
    state, end, _ = env.observe()
    if env.current_player == player_opt.player:
        action_opt = player_opt.act(state)
        state, end, _ = env.step(action_opt)
        
    state = tuple(state.flatten())
    action = pick_action(state, Q_table, eps_agent)

    return player_opt, state, action, end


def print_game(env, game_nb, winner, players):
    """
    Prints the game stats after it ended

    params:
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


def eps_policy(env, nb_epochs=20000, eps_agent=0., eps_opt=0.5, alpha=0.5, gamma=0.99, print_last_10_games=False):
    """
    Trains an agent to play Tic Tac Toe using an epsilon-greedy policy. 
    See section 6.5 of https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf

    params:
        env: Instance of the TictactoeEnv class from ``tic_env.py``
        nb_epochs: Number of games played
        eps_agent: Probability of the agent to make a random move
        eps_opt: Probability of the optimal player to make a random move
        alpha: Learning rate, value between 0 and 1
        gamma: Discount factor, value between 0 and 1
    """
    #players[0] -> OptimalPlayer
    #players[1] -> Agent
    players = ['X','O']

    Q_table = dict()
    t_start = time.time()

    for epoch in range(nb_epochs):
        player_opt, state, action, end = init_game(env, players, eps_opt, eps_agent, Q_table)
        
        while not end:
            next_state, end, winner = env.step(action)
            
            # Agent wins
            if end:
                update_q(env.reward(players[1]), Q_table, state, action)
                break

            # Optimal player
            action_opt = player_opt.act(next_state)
            next_state, end, winner = env.step(action_opt)
            
            # Optimal player wins
            if end:
                update_q(env.reward(players[1]), Q_table, state, action)
                break
            
            # Agent
            next_state = tuple(next_state.flatten())
            next_action = pick_action(next_state, Q_table, 0)

            # update Q-table using the iterative update rule   
            update_q(env.reward(players[1]), Q_table, state, action, next_state, next_action)

            state = next_state
            action = next_action
        
        
        if print_last_10_games and epoch>=19990:
            print_game(env, epoch, winner, players)

    t_end = time.time()
    print('Learning finished after {:.2f}s\nPlayed a total of {} games'.format((t_end - t_start), nb_epochs))

    return Q_table.copy()