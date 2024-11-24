# CI2024_lab3
The third laboratory requires to solve an extension of the '15 puzzle' with a general nxn dimension. In order to do so, we were asked to use a path search algorithm. Among informed strategies, the most powerful one to find a path from a starting point (random) and a goal state (solution), is A*. Leaving most of the theory details aside, the A* algorithm computes the cost of available moves in the priority queue as the result of two functions: the actual cost and another cost given by an heuristic.   
The first trials I have done for this problem were held by using dijkstra algorithm, which is known to deliver a solution that is the optimal one in terms of cost (actions to go from the initial state to the goal one), that is A* with the heuristic set to return always zero. The problem became clear as soon as I tried to execute it with a square of dimension n=4: the algorithm evaluates too many steps, resulting in a computational cost that doesn't allow to manage it in a reasonable time. It was only useful with the 3x3 instance or lower (requiring a few minutes on my laptop).  
This is the reason why I moved to A*, and first trials were taken using, as heuristic, the Manhattan distance. It worked pretty well, I was able to solve the 3x3 instance in a couple of seconds, the 4x4 instance in some minutes (depending on the starting solution), but still I couldn't find a solution for the 5x5 instance even by letting it run for hours.   
The first thing I thought about was that the algorithm was not strong enough, so I decided to try some variations: IDA* or DDA* (that are reported in trials_lab3.ipynb but not in the delivered solution because not effective as the delivered strategy). The problem, for instance, with the iterative version of A* (IDA*), was that it is actually beneficial for memory (which was not a problem honestly), but it is more prone to end into cycles, so it actually worstened performance. Another implementation I tried is the parallel version of A* using JobLib at the thread level, but the overhead required to manage threads and shared data structures overcomes benefits, so I also left this startegy, but the implementation is still available in the aforementioned file.      
Thus, once I realized it was not possible to aim for the optimum in terms of number of actions required to go from the initial state to the goal for bigger instances, I had to slghtly change my approach. If the problem didn't lie in the algorithm, I had to revise the heuristic. Thus, based on the suggestions received by my collegue Andrea Mirenda (that implemented them), I tried to combine three different heuristics: Manhattand distance, the linear conflict and the walking distance. Here is a brief explanation of the way they work: 
1. **heuristic_manhattan_distance**: This function calculates the sum of the Manhattan distances for each tile from its current position to its goal position (i.e., where it needs to be in the solved puzzle). The Manhattan distance between two points is the sum of the absolute differences of their Cartesian coordinates.

2. **heuristic_linear_conflict**: This function adds an additional penalty to the heuristic estimate if two tiles are in their goal row or column but are reversed relative to their goal order. Each linear conflict adds 2 to the heuristic because two moves are required to resolve the conflict.

3. **heuristic_walking_distance**: Although similar to the Manhattan distance, this heuristic calculates a grid of Manhattan distances for all tiles and sums them up, potentially considering other factors in a full implementation (like blocking or paths).

**compute_multiplication_factor**: This function dynamically adjusts the weight of the heuristic based on the puzzle's dimension, increasing the heuristic's influence for larger puzzles.

**combined_heuristic**: Combines the above heuristics to form a more comprehensive estimate, potentially scaling up based on puzzle size to prefer more accurate but computationally heavier heuristics for larger puzzles.

I report here the code to be more clear with respect to the brief explanation given before:
```python
class PuzzleHeuristicService:
    def __init__(self, goal_state: np.ndarray):
        self.goal_state = goal_state

    def heuristic_manhattan_distance(self, position: np.ndarray) -> int:
        distance = 0
        size = len(position)
        for i in range(size):
            for j in range(size):
                tile = position[i][j]
                if tile != 0:
                    target_row = (tile - 1) // size
                    target_col = (tile - 1) % size
                    distance += abs(i - target_row) + abs(j - target_col)
        return distance

    def heuristic_linear_conflict(self, position: np.ndarray) -> int:
        conflict = 0
        size = len(position)

        # Row conflicts
        for row in range(size):
            max_val = -1
            for col in range(size):
                value = position[row][col]
                if value != 0 and (value - 1) // size == row:
                    if value > max_val:
                        max_val = value
                    else:
                        conflict += 2

        # Column conflicts
        for col in range(size):
            max_val = -1
            for row in range(size):
                value = position[row][col]
                if value != 0 and (value - 1) % size == col:
                    if value > max_val:
                        max_val = value
                    else:
                        conflict += 2

        return conflict

    def heuristic_walking_distance(self, position: np.ndarray) -> int:
        # Calculate the Manhattan distance grid
        size = len(position)
        distance_grid = [[0] * size for _ in range(size)]

        for row in range(size):
            for col in range(size):
                value = position[row][col]
                if value != 0:
                    target_row = (value - 1) // size
                    target_col = (value - 1) % size
                    distance_grid[row][col] = abs(row - target_row) + abs(col - target_col)

        # Sum the distances
        walking_distance = sum(sum(row) for row in distance_grid)
        return walking_distance
    def compute_multiplication_factor(self) -> int:
        if PUZZLE_DIM <= 5:
            return 1
        elif PUZZLE_DIM >= 6:
            return 5

    def combined_heuristic(self, position: np.ndarray) -> int:
        if PUZZLE_DIM <= 3:
            return  self.heuristic_manhattan_distance(position)
        else:
            return self.compute_multiplication_factor() * (
                self.heuristic_manhattan_distance(position)
                + self.heuristic_linear_conflict(position)
                + self.heuristic_walking_distance(position)
            )
```  
There is one detail I proposed to add with my collegue and that gave many improvements in terms of the instances we were able to solve. In particular, by summing the three heuristics, there are cases in which the heuristic itself is overestimating the length of the path to the solution for a minimization problem. Thus, it is said it is not admissible, It is not guaranteed to find the optimum anymore. Anyway, this decreases a lot the number of states evaluated, making feasible to find a solution in a reasonable time for larger instances. By making some trials, we were able to fine-tune parameters in the 'compute_multiplication_factor' function in order to solve up to the 6x6 instance. Out of fairness, I have to say that there are cases in which the algorithm is able to solve also the 7x7 instance in a reasonable time, but it doesn't always happen as it does in the other cases, it depends on the starting point. For instances that has n<=3, I decided to keep one single heuristic among the three, in a way that we are able to find the optimum (since it is small, it does not require too much time). It was not easy to decide what to do in this case: the trade-off between the cost of the algorithm and the quality of the solution made it difficult to decide. Using the three heuristic together with a multiplication factor=1, we are able to find solutions that are close to the optimum on average with a few states evaluated. I couldn't decide which one among them to keep, so I leave the original implementation in this case, leaving the changes for the others making them solvable, even if the optimum, on avergae, is not found. I am not able to empirically verify the last statement for instances bigger than 4, I didn't find an algorithm that guarantees the optimum able to solve it in a reasonable time. Here I also report the code for the A* algorithm, refined in order to use efficiently data structures through an heap and to keep memory of visited states through a set where states are converted to bytes in order to be stored. Hashing, in my case, gave worst result in terms of performance and the tobytes() function is working very well for the purpose. 
```python
def solve_with_enhanced_a_star(initial_state: np.ndarray, goal_state: np.ndarray) -> tuple[list, int]:
    heuristic_service = PuzzleHeuristicService(goal_state)

    def calculate_heuristic(state: np.ndarray) -> int:
        return heuristic_service.combined_heuristic(state)

    # Priority queue: (f_score, g_score, state_bytes, path)
    open_set = []
    heappush(open_set, (calculate_heuristic(initial_state), 0, initial_state.tobytes(), []))
    visited = set()
    goal_state_bytes = goal_state.tobytes()

    counter_action_evaluated = 0

    while open_set:
        # Extract the node with the lowest f_score
        f_score, g_score, current_bytes, path = heappop(open_set)
        current_state = np.frombuffer(current_bytes, dtype=initial_state.dtype).reshape(initial_state.shape)

        # Check if we've reached the goal state
        if current_bytes == goal_state_bytes:
            return path, counter_action_evaluated

        # Add current state to visited
        visited.add(current_bytes)

        # Generate all possible moves
        for act in available_actions(current_state):
            counter_action_evaluated += 1
            next_state = do_action(current_state, act)
            next_bytes = next_state.tobytes()

            if next_bytes in visited:
                continue

            # Update scores
            new_g_score = g_score + 1
            new_f_score = new_g_score + calculate_heuristic(next_state)

            # Add new state to open set
            heappush(open_set, (new_f_score, new_g_score, next_bytes, path + [act]))

    return None, counter_action_evaluated  # No solution found

goal_state = np.array([i for i in range(1, PUZZLE_DIM**2)] + [0]).reshape((PUZZLE_DIM, PUZZLE_DIM))
path, evaluated_states = solve_with_enhanced_a_star(state, goal_state)
print("Path to solution:", path)
print("Number of states evaluated:", evaluated_states)
print("Goodness of the solution: "  + str(len(path)) + " moves")
```    
In order to report results, I decided to execute the algorithm for n>=2 and n<8 5 times for each instance and keeping the best result obtained in terms of number of actions needed to reach the soltion (quality). In general, since we are randomizing a lot the starting point with the function provided by the professor, there is a lot of variability among different trials both in terms of quality and cost. Here is a summary of the performance of the algorithm:

| Dimension | Quality (number of actions from starting to goal state) | Cost (number of actions evaluated) |
|-----------|----------------------------------------------------------|------------------------------------|
| 2         | 2                                                        | 4                                  |
| 3         | 22                                                       | 2936                               |
| 4         | 50                                                       | 29159                              |
| 5         | 138                                                      | 273750                             |
| 6         | 382                                                      | 47837                              |
| 7         | 790                                                      | 1139539                            |  


Resuming what I was saying before with respect to the 7x7 instance, I let the algorithm run till the end the five instances and it was always able to find a solution. Out of fairness, again, there have been times in which the execution took approximately 10 minutes on my laptop evaluating more than 3 millions states.