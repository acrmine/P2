from mcts_node import MCTSNode
from p2_t3 import Board
import random
import math

num_nodes = 900
explore_faction = 2.

last_good_replies = {}

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree until the end criterion are met.
    e.g. find the best expandable node (node with untried action) if it exist,
    or else a terminal node

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        bot_identity:   The bot's identity, either 1 or 2

    Returns:
        node: A node from which the next stage of the search can proceed.
        state: The state associated with that node

    """
    if len(node.untried_actions) > 0:
        return node, state

    best_node = (0, None)
    for c_node in node.child_nodes.values():
        opponent = (board.current_player(state) != bot_identity)
        val = ucb(c_node, opponent)
        if val >= best_node[0]:
            best_node = (val, c_node)

    if best_node[1] is not None:
        best_state = board.next_state(state, best_node[1].parent_action)
        return traverse_nodes(best_node[1], board, best_state, bot_identity)

    return node, state


def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    if not board.is_ended(state) and len(node.untried_actions) > 0:
        new_action = random.choice(node.untried_actions)
        node.untried_actions.remove(new_action)
        new_state = board.next_state(state, new_action)
        new_legal_actions = board.legal_actions(new_state)
        new_node = MCTSNode(node, new_action, new_legal_actions)
        node.child_nodes.update({new_action: new_node})
    else:
        return node, state
    return new_node, new_state


def rollout(board: Board, state, bot_identity: int):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
        bot_identity: Integer representing the bot's identity (1 or 2).

    Returns:
        state: The terminal game state

    """
    action_used = []
    prev_action = None
    while not board.is_ended(state):
        if prev_action is not None and prev_action in last_good_replies:
            new_action = last_good_replies[prev_action]
            if new_action not in board.legal_actions(state):
                new_action = random.choice(board.legal_actions(state))
        else:
            new_action = random.choice(board.legal_actions(state))
        next_state = board.next_state(state, new_action)
        if next_state is None:
            break
        state = next_state
        action_used.append((new_action, board.current_player(state)))
        prev_action = new_action

    we_won = is_win(board, state, bot_identity)
    winning_bot = bot_identity if we_won else bot_identity + 1
    winning_bot = winning_bot if winning_bot <= 2 else 1

    while len(action_used) != 0:
        curr_action = action_used.pop()
        if curr_action[1] != winning_bot:
            continue
        if len(action_used) == 0:
            break
        preceding_action = action_used.pop()
        last_good_replies[preceding_action[0]] = curr_action[0]

    return state, we_won



def backpropagate(node: MCTSNode | None, won: bool):
    """ Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.

    Args:
        node:   A leaf node.
        won:    An indicator of whether the bot won or lost the game.

    """
    while node is not None:
        if won:
            node.wins += 1
        node.visits += 1
        node = node.parent
    return


def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot

    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot
    Returns:
        The value of the UCB function for the given node
    """
    ucb_val = node.wins / node.visits
    if is_opponent:
        ucb_val = 1 - ucb_val

    ucb_val += explore_faction * math.sqrt(math.log(node.parent.visits) / node.visits)

    return ucb_val


def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node

    """
    best_pick = (0, None)
    for c_node in root_node.child_nodes.values():
        val = c_node.wins / c_node.visits
        if val >= best_pick[0]:
            best_pick = (val, c_node)

    return best_pick[1].parent_action


def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1


def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state)  # 1 or 2
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    for _ in range(num_nodes):
        state = current_state
        node = root_node
        # Do MCTS - This is all you!
        # ...
        node, state = traverse_nodes(node, board, state, bot_identity)
        node, state = expand_leaf(node, board, state)

        rollout_state, win = rollout(board, state, bot_identity)
        backpropagate(node, win)

    # Return an action, typically the most frequently used action (from the root) or the action with the best
    # estimated win rate.
    best_action = get_best_action(root_node)

    print(f"(Bot {bot_identity}(modified)) Action chosen: {best_action}")
    return best_action
