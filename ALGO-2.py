import heapq
import random
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import sys

REFERENCE_SOLUTION = [0, 4, 7, 5, 2, 6, 1, 3]

def is_valid(state):
    """
    Check if 'state' is a valid configuration of queens.
    'state' is a list of length 8, where index = row, value = column.
    No two queens can attack each other => no shared columns or diagonals.
    Returns: True if valid, False otherwise.
    """
    n = len(state)
    for r1 in range(n):
        c1 = state[r1]
        for r2 in range(r1 + 1, n):
            c2 = state[r2]
            if c1 == c2: return False
            if abs(r1 - r2) == abs(c1 - c2): return False
    return True

def is_goal(state):
    """
    Goal check: A valid state with no conflicts is a solution.
    Args: state (list): Current state representation.
    Returns: True if it's a goal state, False otherwise.
    """
    return is_valid(state)

def generate_successors(state):
    """
    Generate all possible next states by moving one queen in its row
    to a different column.
    Args: state (list): Current state representation.
    Returns: list: A list of possible successor states.
    """
    successors = []
    n = len(state)
    for row in range(n):
        original_col = state[row]
        for new_col in range(n):
            if new_col != original_col:
                new_state = list(state)
                new_state[row] = new_col
                successors.append(new_state)
    return successors

def heuristic_f3(state, reference=REFERENCE_SOLUTION):
    """
    Calculate F3 heuristic: number of misplaced queens vs reference.
    Args:
        state (list): Current state representation.
        reference (list, optional): The reference solution state. Defaults to REFERENCE_SOLUTION.
    Returns: int: The number of misplaced queens.
    """
    mismatch = 0
    for row in range(len(state)):
        if state[row] != reference[row]:
            mismatch += 1
    return mismatch

def bfs(start_state):
    """
    Breadth-First Search for the 8-Queens problem.
    Args: start_state (list): The initial state.
    Returns: dict: A dictionary containing search results ('solution', 'steps', etc.)
                   'solution' is None if no solution is found.
    """
    queue = deque([start_state])
    visited = set()
    visited.add(tuple(start_state))
    steps = 0
    nodes_yielding_no_new_states = 0
    true_dead_ends = 0
    generated = 0
    max_stored = 1

    while queue:
        current = queue.popleft()
        steps += 1
        if is_goal(current):
            return {'solution': current, 'steps': steps, 'nodes_yielding_no_new_states': nodes_yielding_no_new_states, 'true_dead_ends': true_dead_ends, 'generated': generated, 'max_stored': max_stored}

        successors = generate_successors(current)
        generated += len(successors)
        if not successors: true_dead_ends += 1

        valid_successors = []
        for succ in successors:
            if tuple(succ) not in visited:
                valid_successors.append(succ)
                visited.add(tuple(succ))

        if not valid_successors: nodes_yielding_no_new_states += 1

        for succ in valid_successors: queue.append(succ)
        max_stored = max(max_stored, len(queue) + len(visited))

    return {'solution': None, 'steps': steps, 'nodes_yielding_no_new_states': nodes_yielding_no_new_states, 'true_dead_ends': true_dead_ends, 'generated': generated, 'max_stored': max_stored}

def a_star(start_state):
    """
    A* Search for the 8-Queens problem, using the heuristic F3.
    Args: start_state (list): The initial state.
    Returns: dict: A dictionary containing search results ('solution', 'steps', etc.)
                   'solution' is None if no solution is found.
    """
    open_list = []
    heapq.heappush(open_list, (heuristic_f3(start_state, REFERENCE_SOLUTION), start_state, 0))
    closed_set = {tuple(start_state)}
    steps = 0
    nodes_yielding_no_new_states = 0
    true_dead_ends = 0
    generated = 0
    max_stored = 1

    while open_list:
        f_val, current, g_val = heapq.heappop(open_list)
        steps += 1
        if is_goal(current):
            return {'solution': current, 'steps': steps, 'nodes_yielding_no_new_states': nodes_yielding_no_new_states, 'true_dead_ends': true_dead_ends, 'generated': generated, 'max_stored': max_stored}

        successors = generate_successors(current)
        generated += len(successors)
        if not successors: true_dead_ends += 1

        valid_successors_to_push = []
        for succ in successors:
            succ_tup = tuple(succ)
            if succ_tup not in closed_set:
                closed_set.add(succ_tup)
                valid_successors_to_push.append(succ)

        if not valid_successors_to_push: nodes_yielding_no_new_states += 1

        for succ in valid_successors_to_push:
            g_new = g_val + 1
            h_new = heuristic_f3(succ, REFERENCE_SOLUTION)
            f_new = g_new + h_new
            heapq.heappush(open_list, (f_new, succ, g_new))
        max_stored = max(max_stored, len(open_list) + len(closed_set))

    return {'solution': None, 'steps': steps, 'nodes_yielding_no_new_states': nodes_yielding_no_new_states, 'true_dead_ends': true_dead_ends, 'generated': generated, 'max_stored': max_stored}

def generate_initial_state():
    """
    Generates a random initial state for the 8-Queens problem.
    Returns: list: A randomly shuffled list representing queen positions.
    """
    state = list(range(8))
    random.shuffle(state)
    return state

def draw_board(state, title="8-Queens State"):
    """
    Draws the chessboard with queens based on the state list.
    Args:
        state (list): The state representing queen positions.
        title (str, optional): The title for the plot window. Defaults to "8-Queens State".
    """
    n = len(state)
    light_brown = '#DEB887' # burlywood
    dark_brown = '#8B4513'  # saddlebrown
    cmap = mcolors.ListedColormap([light_brown, dark_brown])

    board_pattern = np.zeros((n, n))
    board_pattern[1::2, ::2] = 1
    board_pattern[::2, 1::2] = 1

    fig, ax = plt.subplots()
    ax.imshow(board_pattern, cmap=cmap, extent=[0, n, 0, n])

    for r in range(n):
        c = state[r]
        ax.text(c + 0.5, n - 1 - r + 0.5, u'\u265B', fontsize=20, ha='center', va='center', color='black')

    ax.set_xticks(np.arange(n + 1))
    ax.set_yticks(np.arange(n + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(n + 1), minor=True)
    ax.set_yticks(np.arange(n + 1), minor=True)
    ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
    ax.set_title(title)
    ax.invert_yaxis()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

def process_and_display_result(algo_name, result_dict):
    """
    Prints results and visualizes the board if solution exists.
    Args:
        algo_name (str): The name of the algorithm (e.g., "BFS", "A*").
        result_dict (dict): The dictionary returned by the search algorithm.
    """
    print(f"\n---------- {algo_name} Results ----------")
    if result_dict and result_dict.get('solution'):
        solution_state = result_dict['solution']
        print("  Status:                Solution Found!")
        print(f"  Final State:           {solution_state}")
        print(f"  Steps Taken:           {result_dict.get('steps', 'N/A'):<10}")
        print(f"  States Generated:      {result_dict.get('generated', 'N/A'):<10}")
        print(f"  Nodes (No New States): {result_dict.get('nodes_yielding_no_new_states', 'N/A'):<10}")
        print(f"  Nodes (True Dead Ends):{result_dict.get('true_dead_ends', 'N/A'):<10} (Expected 0 for this problem)")
        print(f"  Max Memory (Nodes):    {result_dict.get('max_stored', 'N/A'):<10}")
        print(f"\n  Displaying {algo_name} solution board (close plot window to continue)...")
        draw_board(solution_state, title=f"{algo_name} Solution")
    else:
        print("  Status:                No Solution Found.")
        if result_dict:
            print(f"  Steps Taken:           {result_dict.get('steps', 'N/A'):<10}")
            print(f"  States Generated:      {result_dict.get('generated', 'N/A'):<10}")
            print(f"  Nodes (No New States): {result_dict.get('nodes_yielding_no_new_states', 'N/A'):<10}")
            print(f"  Nodes (True Dead Ends):{result_dict.get('true_dead_ends', 'N/A'):<10} (Expected 0 for this problem)")
            print(f"  Max Memory (Nodes):    {result_dict.get('max_stored', 'N/A'):<10}")
        else:
             print("  Error: Result dictionary is missing or algorithm failed unexpectedly.")
    print(f"--------------------------------------")


if __name__ == "__main__":

    initial_state = None
    while True:
        choice = input("Choose initial state: (1) Random, (2) Manual input: ").strip()
        if choice == '1':
            initial_state = generate_initial_state()
            print(f"\nGenerated random state: {initial_state}")
            break
        elif choice == '2':
            state_str = input("Enter initial state as a list of 8 column numbers (e.g., [0, 4, 7, 5, 2, 6, 1, 3]): ")
            try:
                state_list = eval(state_str)
                if isinstance(state_list, list) and len(state_list) == 8:
                    if all(isinstance(x, int) and 0 <= x < 8 for x in state_list) and len(set(state_list)) == 8:
                        initial_state = state_list
                        print(f"\nUsing manual state: {initial_state}")
                        break
                    else:
                        print("Invalid state: List must contain 8 unique integers between 0 and 7.")
                else:
                    print("Invalid input format. Ensure it's a list of 8 numbers.")
            except Exception as e:
                print(f"Error parsing input: {e}. Please use Python list format, e.g., [0, 1, 2, 3, 4, 5, 6, 7]")
        else:
            print("Invalid choice. Please enter 1 or 2.")

    if initial_state is None:
      print("Failed to set initial state. Exiting.")
      sys.exit(1)

    print("\nDisplaying initial board (close plot window to continue)...")
    draw_board(initial_state, title="Initial State")

    algo_choice = None
    while True:
        choice = input("\nChoose algorithm(s) to run: (1) BFS only, (2) A* only, (3) Both: ").strip()
        if choice in ['1', '2', '3']:
            algo_choice = choice
            break
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

    bfs_result = None
    astar_result = None

    if algo_choice == '1' or algo_choice == '3':
        print("\nRunning BFS search...")
        bfs_result = bfs(initial_state)
        process_and_display_result("BFS", bfs_result)

    if algo_choice == '2' or algo_choice == '3':
        if algo_choice == '3': print("\n")
        print("Running A* search...")
        astar_result = a_star(initial_state)
        process_and_display_result("A*", astar_result)

    print("\nExecution finished.")