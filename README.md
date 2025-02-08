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

# Review Lab 3

For what concerns the review process of lab3, I was assigned two repositories of two collegues. I will report my review for both of them.  
The first review is done for the code delivered by MarilenaDel.  
She delivered a well structured version of A* algorithm, specifically using the Manhattan distance as heuristic function. Here is my review for her code:  

``` 
First of all, I would like to highlight the clairity and the structure of the code. It was really easy to go through it and understand all the details.

Then, the implementation of the A* algorithm seems correct to me and it is also efficient making use of priority queues (heapq) to manage the open list and a set for the closed set, ensuring quick lookups and preventing redundant state evaluation. The use of tobytes for storing state representations is a clever way to handle immutability and efficient comparison in the closed set.

Thus, I have nothing to report with respect to the implementation of the A* algorithm and the chosen strategy in general, as suggested by the professor, this is the best informed strategy when dealing with path search.

Therefore, the only enhancement I can suggest is related to the heuristic you decided to use. As you stated in the readme, it is admissible, thus it always leads to the shortest path, but yet it evaluates a huge number of states. Even if you didn't report the result obtained with your algorithm, I suppose (from my experiments) it is not able to find a solution in a reasonable time for instances larger than 3. In order to make it feasible, I suggest you to change the heuristic when the square grows. In particular, when the heuristic overestimates the length of the path, the number of evaluated states decreases a lot!

For instance, if you want to have a look at my code, I combined three different heuristics: manhattan (that you used as well), linear conflict and walking distance. I used this approach only for n>=4 and, thanks to an additional multiplication factor, I was able to solve squares up to dimension 6 (unfortunately, the solution is not the optimal one.. but you can't get everything!).

The approach you used is the same I used for instances with n<=3, so I perfectly agree with the choice. I hope that my suggestions could be useful to enhance even more the current performance of your A* algorithm.

```   
Since I found the solution very well structured and efficient, I only tried to propose an enhancement to give her the possibility to solve greater instances.   


For what concerns the other review, it was done for the code delivered by yuripettorossi. Here is my review:  

``` 
I tried to go through the code of the delivered solution, I have to say that it took a lot of time to do it because there are a lot of unused functions and duplicates, that result in many lines of code not easy to be read. Anyway, I tried both the strategies that you delivered, even if my code was slightly different. If you want to keep focusing on that strategies (BFS and Bi-Directional Search), the only suggestion I can give you is to use memory in a more efficient way. You are currently using a list of lists to store already visited states, which is really costly and expensive also from the computational point of view. Try instead to use tobytes to hash already visited states, it is faster!

However, to be honest, I would suggest you to go back to the A* that you left in 'tests' and re-start from that. The same suggestion I gave you for the BFS can be applied to the A* and, enhancing it with an heap to store not only states but also the costs, is even better!

Now, in the provided solution, you are using an approach that does not allow you to find the optimal solution even for smaller instances, but still evaluates a large number of states. You can do something similar with A*, obtaining better results (closer to the optimum) and with a lower number of states evaluated. Indeed, if the heuristic is not admissible anymore, and you start overestimating for a minimization problem like this, the number of evaluated states gets lower. Try to combine more than one heuristic in the A*. I have seen you are currently using the manhattan distance, try to sum it to some other heuristic (walking distance and so on...). By overestimating, obviously you are not finding the optimum anymore, but, in any case, the same happens with BFS, but you should obtain something better both in terms of execution times and cost.

I hope that my suggestions will be useful to enhance the performance of your algorithms!
```   

In this case I tried to suggest another strategy with respect to the two tried by the author (that are reported in the review itself). I tried both of them and then changed approach because they were not able so solve even very small instances in a reasonable time. Moreover, in my opinion, the usage of data structures was not efficient, which can result in even more time in order to perform exectuion. So, I wouldn't know how to make them better rather than the suggestion I gave. Thus, as done in class, I tried to propose another algorithm to make it better.  

# Review for my code of Lab 3  

Here are the two reviews I received for my code, I really appreciated the comments of my collegues, even if there were no suggestions to further enhance the algorithm (except for a suggestion to report results in the README in a different way).  

```  
The structure of the repository is very clear and it helps in understanding the implementation of the code.

I will not review the implementation of the code, since your results were better than mine, both in term of quality and dimension of puzzle solved, and I would not have anything to suggest to help you improve the performances.

I think the implementation of different heuristics, to reduce the number of visited states, to find the solution without wasting resources, is the key part of your work. I will definitely try to implement some of those heuristics in my project, as you suggested in your review.

In general, I find your work very good.
The only small detail I would change is related to the results.
Since you have already simulated 5 different solves, it would may be a good idea to compute the average of the different solutions, to give a more general overview of the problem, rather than the best case scenario.
Or even compute the average leaving out the best and the worst case, like in the official puzzle-solving competitions.
As I mentioned, it is just a minor change, but since your solution was already very good, it may interest you.


```     
First of all, I have to thank you for the exhaustive description in the readme. Everything is clear and I don't even need to look at the code trying to decifrate it 🤣 .
I went through a similar process: using admissible heuristics was unfeasible for big instances of the problem, so I also found a way to violate the constraint and still get a good result (I elevated the Manhattan distance to an exponent that gest gradually bigger as the number of evaluated states grows, you can find it here).
Out of the three heuristics you used, I find heuristic_linear_conflict really interesting as I never thought of it during my attempts to find new heuristics.
The usage of the byte representation instead of hashing is certainly one thing that I can copy in my implementation to speed it up without changing the structure of the code. Thanks.
```  




```    





