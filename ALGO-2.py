import heapq
import random

REFERENCE_SOLUTION = [0, 4, 7, 5, 2, 6, 1, 3]



def is_valid(state):
    """
    Check if 'state' is a valid configuration of queens.
    'state' is a list of length 8, where index = row, value = column.
    No two queens can attack each other => no shared columns or diagonals.
    """
    n = len(state)
    for r1 in range(n):
        c1 = state[r1]
        for r2 in range(r1 + 1, n):
            c2 = state[r2]
            if c1 == c2:  # same column
                return False
            if abs(r1 - r2) == abs(c1 - c2):  # same diagonal
                return False
    return True


def is_goal(state):
    """Goal check: A valid state with no conflicts is a solution."""
    return is_valid(state)


def generate_successors(state):
    """
    Generate all possible next states by moving one queen in its row
    to a different column.
    """
    successors = []
    n = len(state)
    for row in range(n):
        original_col = state[row]
        for new_col in range(n):
            # Skip the same col to avoid duplicating the same state
            if new_col != original_col:
                new_state = list(state)
                new_state[row] = new_col
                successors.append(new_state)
    return successors


def heuristic_f3(state, reference=REFERENCE_SOLUTION):
    mismatch = 0
    for row in range(len(state)):
        if state[row] != reference[row]:
            mismatch += 1
    return mismatch



def bfs(start_state):
    """
    Breadth-First Search for the 8-Queens problem.

    Returns a dictionary with:
      - 'solution': the first found solution state
      - 'steps': number of expansions performed
      - 'dead_ends': number of states where no valid successors
      - 'generated': total states generated
      - 'max_stored': peak number of states stored in memory
    """
    from collections import deque

    queue = deque([start_state])
    visited = set()
    visited.add(tuple(start_state))

    steps = 0
    dead_ends = 0
    generated = 0
    max_stored = 1

    while queue:
        current = queue.popleft()
        steps += 1

        if is_goal(current):
            return {
                'solution': current,
                'steps': steps,
                'dead_ends': dead_ends,
                'generated': generated,
                'max_stored': max_stored
            }

        successors = generate_successors(current)
        valid_successors = []
        for succ in successors:
            generated += 1
            if tuple(succ) not in visited:
                valid_successors.append(succ)
                visited.add(tuple(succ))

        if not valid_successors:
            dead_ends += 1

        for succ in valid_successors:
            queue.append(succ)

        max_stored = max(max_stored, len(queue) + len(visited))

    return {
        'solution': None,
        'steps': steps,
        'dead_ends': dead_ends,
        'generated': generated,
        'max_stored': max_stored
    }


def a_star(start_state):
    """
    A* Search for the 8-Queens problem, using the heuristic F3.

    Returns a dictionary with:
      - 'solution': the found solution state
      - 'steps': number of expansions performed
      - 'dead_ends': number of states where no valid successors
      - 'generated': total states generated
      - 'max_stored': peak number of states stored in memory
    """
    open_list = []
    heapq.heappush(open_list, (heuristic_f3(start_state), start_state, 0))  # (f, state, g)
    closed_set = {tuple(start_state)}

    steps = 0
    dead_ends = 0
    generated = 0
    max_stored = 1

    while open_list:
        f_val, current, g_val = heapq.heappop(open_list)
        steps += 1

        if is_goal(current):
            return {
                'solution': current,
                'steps': steps,
                'dead_ends': dead_ends,
                'generated': generated,
                'max_stored': max_stored
            }

        # Generate successors
        successors = generate_successors(current)
        valid_successors = []
        for succ in successors:
            generated += 1
            succ_tup = tuple(succ)
            if succ_tup not in closed_set:
                valid_successors.append(succ)
                closed_set.add(succ_tup)

        if not valid_successors:
            dead_ends += 1

        for succ in valid_successors:
            g_new = g_val + 1
            h_new = heuristic_f3(succ, REFERENCE_SOLUTION)
            f_new = g_new + h_new
            heapq.heappush(open_list, (f_new, succ, g_new))

        max_stored = max(max_stored, len(open_list) + len(closed_set))

    # If somehow no solution found
    return {
        'solution': None,
        'steps': steps,
        'dead_ends': dead_ends,
        'generated': generated,
        'max_stored': max_stored
    }



def generate_initial_state():
    state = list(range(8))  # [0,1,2,3,4,5,6,7]
    random.shuffle(state)
    return state


def run_experiments(num_experiments=40):
    results = []

    for i in range(num_experiments):
        init_state = generate_initial_state()

        # BFS
        bfs_result = bfs(init_state)
        # A*
        astar_result = a_star(init_state)

        # Save everything in a row
        row = {
            'Initial': init_state,
            'BFS_Steps': bfs_result['steps'],
            'A*_Steps': astar_result['steps'],
            'BFS_DeadEnds': bfs_result['dead_ends'],
            'A*_DeadEnds': astar_result['dead_ends'],
            'BFS_Generated': bfs_result['generated'],
            'A*_Generated': astar_result['generated'],
            'BFS_Mem': bfs_result['max_stored'],
            'A*_Mem': astar_result['max_stored'],
        }
        results.append(row)

    print("\n{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
        "Initial State",
        "BFS Steps",
        "A* Steps",
        "BFS DeadEnds",
        "A* DeadEnds",
        "BFS Generated",
        "A* Generated",
        "BFS Mem",
        "A* Mem"
    ))
    print("-" * 190)

    for row in results:
        print("{:<30} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20} | {:<20}".format(
            str(row['Initial']),
            row['BFS_Steps'],
            row['A*_Steps'],
            row['BFS_DeadEnds'],
            row['A*_DeadEnds'],
            row['BFS_Generated'],
            row['A*_Generated'],
            row['BFS_Mem'],
            row['A*_Mem']
        ))

    return results



if __name__ == "__main__":
    final_results = run_experiments(num_experiments=40)

    bfs_steps = sum(r['BFS_Steps'] for r in final_results) / len(final_results)
    a_star_steps = sum(r['A*_Steps'] for r in final_results) / len(final_results)

    print("\nAverage BFS steps:", bfs_steps)
    print("Average A* steps: ", a_star_steps)
