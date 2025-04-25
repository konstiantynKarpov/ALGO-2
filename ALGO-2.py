import heapq
import random
from collections import deque
import tkinter as tk
from tkinter import messagebox
import sys

REFERENCE_SOLUTION = [0, 4, 7, 5, 2, 6, 1, 3]

def is_valid(state):
    """
    Checks if the current placement of queens in the state is valid.
    Considers only placed queens (value != -1).
    Args: state (list): The current state representation.
    Returns: True if valid, False otherwise.
    """
    n = len(state)
    placed_queens = []
    for r, c in enumerate(state):
        if c != -1:
            placed_queens.append((r, c))

    if len(placed_queens) <= 1:
        return True

    for i in range(len(placed_queens)):
        r1, c1 = placed_queens[i]
        for j in range(i + 1, len(placed_queens)):
            r2, c2 = placed_queens[j]
            if c1 == c2: return False
            if abs(r1 - r2) == abs(c1 - c2): return False
    return True

def is_goal(state):
    """
    Checks if the state represents a valid solution (8 non-attacking queens).
    Requires all queens to be placed (no -1 values).
    Args: state (list): The state to check.
    Returns: True if it's a goal state, False otherwise.
    """
    if any(col == -1 for col in state):
        return False
    return is_valid(state)

def generate_successors(state):
    """
    Generates successor states ONLY from a complete state (no -1 values).
    Args: state (list): A complete state representation.
    Returns: list: A list of possible successor states. Returns empty list if state is incomplete.
    """
    if any(col == -1 for col in state):
        return []
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
    Calculates heuristic based on mismatches with REFERENCE_SOLUTION.
    Considers unplaced queens (-1) as mismatches.
    Args:
        state (list): Current state representation.
        reference (list, optional): The reference solution state. Defaults to REFERENCE_SOLUTION.
    Returns: int: The number of misplaced/unplaced queens.
    """
    mismatch = 0
    n = len(state)
    for row in range(n):
        if state[row] == -1 or state[row] != reference[row]:
            mismatch += 1
    return mismatch

def bfs(start_state):
    """
    Breadth-First Search for the 8-Queens problem. Requires a complete start state.
    Args: start_state (list): The initial complete state.
    Returns: dict: A dictionary containing search results ('solution', 'steps', etc.)
                   'solution' is None if no solution is found or start state is invalid.
    """
    if any(col == -1 for col in start_state):
        print("BFS Error: Cannot start BFS with an incomplete state.")
        return {'solution': None, 'steps': 0, 'nodes_yielding_no_new_states': 0, 'true_dead_ends': 0, 'generated': 0, 'max_stored': 0}
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

        valid_successors = []
        for succ in successors:
            if tuple(succ) not in visited:
                valid_successors.append(succ)
                visited.add(tuple(succ))
        if not valid_successors and successors: nodes_yielding_no_new_states += 1
        for succ in valid_successors: queue.append(succ)
        max_stored = max(max_stored, len(queue) + len(visited))
    return {'solution': None, 'steps': steps, 'nodes_yielding_no_new_states': nodes_yielding_no_new_states, 'true_dead_ends': true_dead_ends, 'generated': generated, 'max_stored': max_stored}

def a_star(start_state):
    """
    A* Search for the 8-Queens problem. Requires a complete start state.
    Args: start_state (list): The initial complete state.
    Returns: dict: A dictionary containing search results ('solution', 'steps', etc.)
                   'solution' is None if no solution is found or start state is invalid.
    """
    if any(col == -1 for col in start_state):
        print("A* Error: Cannot start A* with an incomplete state.")
        return {'solution': None, 'steps': 0, 'nodes_yielding_no_new_states': 0, 'true_dead_ends': 0, 'generated': 0, 'max_stored': 0}
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

        valid_successors_to_push = []
        for succ in successors:
            succ_tup = tuple(succ)
            if succ_tup not in closed_set:
                closed_set.add(succ_tup)
                valid_successors_to_push.append(succ)
        if not valid_successors_to_push and successors: nodes_yielding_no_new_states += 1
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
    Returns: list: A randomly shuffled list representing a complete state.
    """
    state = list(range(8))
    random.shuffle(state)
    return state

def process_and_display_result(app_instance, algo_name, result_dict):
    """
    Prints results to console and updates the Tkinter board if solution exists.
    Args:
        app_instance (ChessboardApp): The instance of the Tkinter application.
        algo_name (str): The name of the algorithm run (e.g., "BFS").
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
        print(f"  Nodes (True Dead Ends):{result_dict.get('true_dead_ends', 'N/A'):<10} (Expected 0)")
        print(f"  Max Memory (Nodes):    {result_dict.get('max_stored', 'N/A'):<10}")

        app_instance.current_state = solution_state
        app_instance.draw_board()
        app_instance.status_label.config(text=f"{algo_name}: Solution Found!")
        print(f"  Solution displayed on the board.")

    else:
        print("  Status:                No Solution Found.")
        if result_dict:
            print(f"  Steps Taken:           {result_dict.get('steps', 'N/A'):<10}")
            print(f"  States Generated:      {result_dict.get('generated', 'N/A'):<10}")
            print(f"  Nodes (No New States): {result_dict.get('nodes_yielding_no_new_states', 'N/A'):<10}")
            print(f"  Nodes (True Dead Ends):{result_dict.get('true_dead_ends', 'N/A'):<10} (Expected 0)")
            print(f"  Max Memory (Nodes):    {result_dict.get('max_stored', 'N/A'):<10}")
            app_instance.status_label.config(text=f"{algo_name}: No Solution Found.")
        else:
             print("  Error: Result dictionary is missing or algorithm failed unexpectedly.")
             app_instance.status_label.config(text=f"{algo_name}: Error occurred.")
    print(f"--------------------------------------")


class ChessboardApp:
    """Tkinter application for setting up and solving the 8-Queens problem."""
    def __init__(self, master):
        """
        Initializes the Tkinter GUI application.
        Args: master: The root Tkinter window.
        """
        self.master = master
        self.master.title("8-Queens Setup & Solve")
        self.board_size = 8
        self.cell_size = 50
        self.current_state = [-1] * self.board_size
        self.queen_char = u'\u265B'
        self.light_color = '#DEB887'
        self.dark_color = '#8B4513'
        self.queen_color = 'black'

        board_frame = tk.Frame(master)
        board_frame.pack(side=tk.LEFT, padx=10, pady=10)

        canvas_width = self.board_size * self.cell_size
        canvas_height = self.board_size * self.cell_size
        self.canvas = tk.Canvas(board_frame, width=canvas_width, height=canvas_height, borderwidth=0, highlightthickness=0)
        self.canvas.pack()
        self.queen_on_canvas_ids = [None] * self.board_size
        self.canvas.bind("<Button-1>", self.on_click)

        control_frame = tk.Frame(master)
        control_frame.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.Y, expand=True)

        self.status_label = tk.Label(control_frame, text="Click board to place queens (1 per row).\nOr generate random.", justify=tk.LEFT)
        self.status_label.pack(pady=5, anchor='w')

        self.random_button = tk.Button(control_frame, text="Generate Random State", command=self.generate_random_and_draw)
        self.random_button.pack(fill=tk.X, pady=5)

        solve_label = tk.Label(control_frame, text="--- Solve ---")
        solve_label.pack(pady=(10, 0), anchor='w')
        self.solve_bfs_button = tk.Button(control_frame, text="Solve with BFS", command=self.solve_bfs)
        self.solve_bfs_button.pack(fill=tk.X, pady=2)
        self.solve_astar_button = tk.Button(control_frame, text="Solve with A*", command=self.solve_astar)
        self.solve_astar_button.pack(fill=tk.X, pady=2)
        self.solve_both_button = tk.Button(control_frame, text="Solve with Both", command=self.solve_both)
        self.solve_both_button.pack(fill=tk.X, pady=2)

        self.quit_button = tk.Button(control_frame, text="Quit", command=self.master.quit)
        self.quit_button.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

        self.draw_board()

    def draw_board(self):
        """Redraws the queens on the board according to self.current_state."""
        if not self.canvas.find_withtag("board_square"):
            for r in range(self.board_size):
                for c in range(self.board_size):
                    x1, y1 = c * self.cell_size, r * self.cell_size
                    x2, y2 = x1 + self.cell_size, y1 + self.cell_size
                    color = self.light_color if (r + c) % 2 == 0 else self.dark_color
                    self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="black", tags="board_square")
            self.canvas.tag_raise("queen")

        for queen_id in self.queen_on_canvas_ids:
            if queen_id: self.canvas.delete(queen_id)
        self.queen_on_canvas_ids = [None] * self.board_size

        for r in range(self.board_size):
            c = self.current_state[r]
            if c != -1:
                x1, y1 = c * self.cell_size, r * self.cell_size
                center_x, center_y = x1 + self.cell_size / 2, y1 + self.cell_size / 2
                self.queen_on_canvas_ids[r] = self.canvas.create_text(center_x, center_y,
                                                                     text=self.queen_char,
                                                                     font=("Arial", self.cell_size // 2),
                                                                     fill=self.queen_color,
                                                                     tags="queen")

    def on_click(self, event):
        """Handles mouse clicks on the board to place/move a queen in a row."""
        col = event.x // self.cell_size
        row = event.y // self.cell_size

        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            self.current_state[row] = col
            self.draw_board()
            self.status_label.config(text="Queen placed. Click again or Solve...")
        else:
            print("Click outside board ignored.")


    def generate_random_and_draw(self):
        """Generates a random state and displays it on the board."""
        self.current_state = generate_initial_state()
        print(f"\nGenerated random state: {self.current_state}")
        self.draw_board()
        validity = "Valid" if is_valid(self.current_state) else "Invalid"
        self.status_label.config(text=f"Random state ({validity}). Click or Solve...")


    def _run_and_process(self, algorithm_func, algo_name, initial_state):
        """Internal helper to run a search algorithm and process its results."""
        print(f"\nRunning {algo_name} search for state: {initial_state}...")
        self.status_label.config(text=f"Running {algo_name}...")
        self.master.update()
        result_dict = algorithm_func(initial_state)
        process_and_display_result(self, algo_name, result_dict)

    def solve_bfs(self):
        """Checks state and runs BFS."""
        if any(col == -1 for col in self.current_state):
            messagebox.showwarning("Incomplete State", "Please place a queen in every row before solving.")
            return
        self._run_and_process(bfs, "BFS", list(self.current_state))

    def solve_astar(self):
        """Checks state and runs A*."""
        if any(col == -1 for col in self.current_state):
            messagebox.showwarning("Incomplete State", "Please place a queen in every row before solving.")
            return
        self._run_and_process(a_star, "A*", list(self.current_state))

    def solve_both(self):
        """Checks state and runs both BFS and A* sequentially."""
        if any(col == -1 for col in self.current_state):
            messagebox.showwarning("Incomplete State", "Please place a queen in every row before solving.")
            return
        initial_state_copy = list(self.current_state)
        self._run_and_process(bfs, "BFS", initial_state_copy)
        print("\n")
        self._run_and_process(a_star, "A*", initial_state_copy)


if __name__ == "__main__":
    print("Starting 8-Queens GUI...")
    root = tk.Tk()
    app = ChessboardApp(root)
    root.mainloop()
    print("\nGUI closed. Execution finished.")