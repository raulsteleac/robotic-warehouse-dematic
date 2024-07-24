import numpy as np
from gymnasium import spaces

from tarware.definitions import Action, Direction


def flatten_list(l):
    '''
    Flattens a list of lists
    '''
    return [item for sublist in l for item in sublist]

def split_list(lst, n_groups, verbose=False):
    '''
    Splits a list `lst` into `n_groups` groups of approximately equal lengths. 
    This function guarantees that the chunks will be the same length, or differing in length by 1 element at most.
    '''
    # If you divide n elements into roughly k chunks you can make n % k chunks 1 element bigger than the other chunks to distribute the extra elements.
    # https://stackoverflow.com/questions/2130016/splitting-a-list-into-n-parts-of-approximately-equal-length
    k, m = divmod(len(lst), n_groups)
    output = list(lst[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n_groups))
    
    if verbose:
        group_lengths = [len(group) for group in output]
        min_group_length = min(group_lengths)
        max_group_length = max(group_lengths)
        num_groups_with_min_group_length = len([i for i in group_lengths if i == min_group_length])
        num_groups_with_max_group_length = len([i for i in group_lengths if i == max_group_length])

        print(f"Splitting {len(lst)} items into {n_groups} groups")

        if min_group_length == max_group_length:
            print("Even split achieved")
            print(f"Group size = {min_group_length} ({num_groups_with_min_group_length} groups total)")
        else:
            print("Non-even split achieved")
            print(f"Minimum group size = {min_group_length} ({num_groups_with_min_group_length} groups total)")
            print(f"Maximum group size = {max_group_length} ({num_groups_with_max_group_length} groups total)")
    return output

def get_next_micro_action(agent_x, agent_y, agent_direction, target):
    direction_to_enum = {
        (0, -1): Direction.UP,
        (0, 1): Direction.DOWN,
        (-1, 0): Direction.LEFT,
        (1, 0): Direction.RIGHT,
        }
    
    target_x, target_y = target
    target_direction =  direction_to_enum[(target_x - agent_x, target_y - agent_y)]

    turn_order = [Direction.UP, Direction.RIGHT, Direction.DOWN, Direction.LEFT]
    # Find the indices of the source and target directions in the turn order
    source_index = turn_order.index(agent_direction)
    target_index = turn_order.index(target_direction)

    # Calculate the difference in indices to determine the number of turns needed
    turn_difference = (source_index - target_index) % len(turn_order)

    # Determine the direction of the best next turn
    if turn_difference == 0:
        return Action.FORWARD
    elif turn_difference == 1:
        return Action.LEFT
    elif turn_difference == 2:
        return Action.RIGHT
    elif turn_difference == 3:
        return Action.RIGHT

def find_sections(pairs):
    groups = []

    for pair in pairs:
        added = False

        for group in groups:
            if any(abs(pair[0] - gp[0]) + abs(pair[1] - gp[1]) == 1 for gp in group):
                group.append(pair)
                added = True
                break

        if not added:
            groups.append([pair])

    return groups

def comput_valid_action_masks(env, pickers_to_agvs=True, block_conflicting_actions=True):
    requested_items = env.get_shelf_request_information()
    empty_items = env.get_empty_shelf_information()
    carrying_shelf_info = env.get_carrying_shelf_information()
    targets_agvs = [agent.target - len(env.goals) - 1 for agent in env.agents[:env.num_agvs] if agent.target > len(env.goals)] 
    targets_pickers = [agent.target - len(env.goals) - 1 for agent in env.agents[env.num_agvs:] if agent.target > len(env.goals)] 
    # Compute valid location list for AGVs
    valid_location_list_agvs = np.array([
        empty_items if carrying_shelf else requested_items for carrying_shelf in carrying_shelf_info
    ])
    # Compute valid location list for Pickers
    if pickers_to_agvs:
        valid_location_list_pickers = np.zeros(len(env.action_id_to_coords_map) - len(env.goals))
        valid_location_list_pickers[targets_agvs] = 1
    else:
        valid_location_list_pickers = requested_items
    # Mask out conflicting actions for agents of the same type
    if block_conflicting_actions:
        valid_location_list_agvs[:, targets_agvs] = 0
        valid_location_list_pickers[targets_pickers] = 0
    valid_action_masks = np.ones((len(env.action_space), spaces.flatdim(env.action_space[0])))
    valid_action_masks[:env.num_agvs,  1 : 1 + len(env.goals)] = np.repeat(np.expand_dims(np.array(carrying_shelf_info), 1), len(env.goals), axis=1)
    valid_action_masks[:env.num_agvs,  1 + len(env.goals):] = valid_location_list_agvs
    valid_action_masks[env.num_agvs:,  1 : 1 + len(env.goals)] = 0
    valid_action_masks[env.num_agvs:,  1 + len(env.goals):] = valid_location_list_pickers
    return valid_action_masks