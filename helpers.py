import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def render_grid_with_q_values(grid, q_values):
    """
    Returns a string representing the rendered grid

    Params:
        - grid: Game state to be rendered

    Returns:
        A string containing the rendered game state
    """
    value2player = {0: '--', 1: 'X', -1: 'O'}
    rendered_grid = ''
    for i in range(3):
        rendered_grid += '|'
        separator = '    <->    ' if i == 1 else '              '
        for j in range(6):
            if j < 3:
                # Render grid
                rendered_grid += value2player[int(grid[i,j])] 
                if j < 2:
                    rendered_grid += ' '
            else:
                # Render q-values
                i_q = i - 3
                j_q = j - 3
                q_val = '{:.2f}'.format(q_values[i_q, j_q]) if q_values[i_q, j_q] < 0 else ' {:.2f}'.format(q_values[i_q, j_q])
                if j == 3:
                    rendered_grid += '|' + separator + '|' + q_val
                else:
                    rendered_grid += ' ' + q_val
        rendered_grid += '|'
        if i < 2:
            rendered_grid += '\n'

    return rendered_grid


def format_val(value):
    """
    Formats a value replacing the thousands by ```k```
    """
    str_value = ''
    if value > 1000:
        value = int(value/1000)
        value = str(value) + 'k'
    return str_value + str(value)


def plot_grids_heatmap(Q_table, grids, path='./plots/q10'):  
    """
    Plots and saves 
    """
    fig = plt.figure(figsize=(20, 5))
    axes = [fig.add_subplot(131), fig.add_subplot(132), fig.add_subplot(133)]

    for idx, grid in enumerate(grids):
        state = tuple(grid.flatten())
        if sum(state) < 0:
            raise ValueError('An invalid grid has been chosen. Player X -> 1 always starts.')
        if sum(state) > 1:
            raise ValueError('An invalid grid has been chosen. The two players X -> 1 and O -> -1 always play one after the other.')
        q_values = np.ones(grid.shape)*-1
        try:
            for key in Q_table[state]:
                q_values[key] = Q_table[state][key]
            sns.heatmap(q_values, ax=axes[idx], square=True)
            axes[idx].set_title(render_grid_with_q_values(grid, q_values), loc='center')
        except:
            raise ValueError('The game has already been finished or the current state has not been encountered.')

    plt.savefig(path,  bbox_inches='tight')