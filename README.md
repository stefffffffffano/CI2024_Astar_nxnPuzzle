# 🧩 CI2024 – Lab 3: Solving the n×n Sliding Puzzle

📌 **Course**: Computational Intelligence – A.A. 2024/2025  
📌 **Institution**: Politecnico di Torino  
📌 **Topic**: Path search algorithms for solving the generalized *n×n sliding puzzle*  

---

## 🚀 Project Overview  

This project focuses on solving an extended version of the **15-puzzle**, generalized to an arbitrary **n×n dimension**. The goal is to find a sequence of moves that transforms a **random initial configuration** into the **goal state** using an efficient **path search algorithm**.  

Among informed search strategies, the most powerful approach for this problem is **A\***, which balances between **exploration** and **exploitation** by evaluating moves based on:  
- **Actual cost** (number of steps taken).  
- **Estimated cost to goal**, computed via **heuristics**.  

---

## 🔍 Initial Experiments  

### ⏳ **Dijkstra’s Algorithm**
The first approach used was **Dijkstra’s algorithm**, which is equivalent to **A\*** with a **zero heuristic**. While optimal, it quickly became **computationally infeasible** for grids larger than 3×3 due to the sheer number of states evaluated.  

### 🚀 **A\* Algorithm with Manhattan Distance**  
The next step was implementing **A\*** with the **Manhattan distance heuristic**, leading to significant improvements:  
- **3×3 puzzle**: solved in **seconds**.  
- **4×4 puzzle**: solved in **minutes**, depending on the starting configuration.  
- **5×5 puzzle**: computationally impractical, requiring hours or more.  

To improve performance for larger puzzles, additional heuristics were explored.

---

## 🎯 Heuristic Optimization  

### 📌 **Heuristics Used**  
A combination of three heuristics was implemented to enhance A*'s efficiency:  

1. **Manhattan Distance**  
   - Computes the sum of the absolute differences between each tile's current position and its target position.  
   
2. **Linear Conflict**  
   - Adds a penalty when two tiles are in their goal row or column but in the wrong order.  
   - Each conflict requires **at least two moves** to resolve, so the heuristic is adjusted accordingly.  

3. **Walking Distance**  
   - Similar to Manhattan distance but accounts for tile dependencies and blocking factors.  

### 🔥 **Combined Heuristic Strategy**  
For **n ≤ 3**, only the **Manhattan distance** is used to guarantee **optimal solutions**.  
For **n ≥ 4**, the heuristics are **combined** and multiplied by a scaling factor, ensuring:  
✅ **Drastically fewer states evaluated**  
✅ **Faster solutions for larger grids**  
❌ **Not always optimal**, but solutions are close to the best known.  

The function **`compute_multiplication_factor()`** dynamically adjusts heuristic weighting based on puzzle size.  

---

## 🔄 **Enhanced A\* Algorithm Implementation**  

The A* algorithm is optimized to **minimize memory usage** and **maximize efficiency**:  
- **Priority queue (`heapq`)** for efficient state expansion.  
- **Visited states stored as bytes (`tobytes()`)** for fast lookup (instead of hashing).  

### ✨ **Key Enhancements**
✅ **Efficient data structures** – Open list as a **heap**, visited states as **bytes**.  
✅ **State pruning** – Prevents redundant state exploration.  
✅ **Dynamic heuristic scaling** – Reduces computational overhead while maintaining accuracy.  

### 🔎 **A\* Algorithm Implementation**
```python
def solve_with_enhanced_a_star(initial_state: np.ndarray, goal_state: np.ndarray) -> tuple[list, int]:
    heuristic_service = PuzzleHeuristicService(goal_state)

    def calculate_heuristic(state: np.ndarray) -> int:
        return heuristic_service.combined_heuristic(state)

    open_set = []
    heappush(open_set, (calculate_heuristic(initial_state), 0, initial_state.tobytes(), []))
    visited = set()
    goal_state_bytes = goal_state.tobytes()

    counter_action_evaluated = 0

    while open_set:
        f_score, g_score, current_bytes, path = heappop(open_set)
        current_state = np.frombuffer(current_bytes, dtype=initial_state.dtype).reshape(initial_state.shape)

        if current_bytes == goal_state_bytes:
            return path, counter_action_evaluated

        visited.add(current_bytes)

        for act in available_actions(current_state):
            counter_action_evaluated += 1
            next_state = do_action(current_state, act)
            next_bytes = next_state.tobytes()

            if next_bytes in visited:
                continue

            new_g_score = g_score + 1
            new_f_score = new_g_score + calculate_heuristic(next_state)

            heappush(open_set, (new_f_score, new_g_score, next_bytes, path + [act]))

    return None, counter_action_evaluated  # No solution found
```  

## 📊 **Performance Analysis**  

Each experiment was repeated **5 times**, selecting the best solution in terms of **number of moves**. Due to the random nature of starting states, execution time and solution quality vary.  

| **Grid Size** | **Best Solution (Moves)** | **States Evaluated** |
|--------------|---------------------------|----------------------|
| 2×2          | 2 moves                    | 4 states            |
| 3×3          | 22 moves                   | 2,936 states        |
| 4×4          | 50 moves                   | 29,159 states       |
| 5×5          | 138 moves                  | 273,750 states      |
| 6×6          | 382 moves                  | 47,837 states       |
| 7×7          | 790 moves                  | 1,139,539 states    |

### 📝 **Observations**  
- The **7×7 puzzle** was solved in all cases, but execution time varied significantly (up to **10 minutes** in some cases).  
- The **combined heuristic** dramatically reduces the number of states evaluated for **n ≥ 4**, making larger instances feasible.  
- **Trade-off between optimality and execution time**:  
  - For **small puzzles (n ≤ 3)**, the **optimal solution** is guaranteed.  
  - For **larger puzzles (n ≥ 4)**, overestimating heuristics reduces the state space search but may lead to **sub-optimal** solutions.  

---

## 🎯 **Key Findings**  

✅ **Heuristic Combination is Crucial**  
- Using **Manhattan Distance, Linear Conflict, and Walking Distance** together significantly enhances performance.  
- The **multiplication factor** helps scale heuristics effectively for **larger puzzles**.  

✅ **State Encoding Improves Efficiency**  
- **Storing states as bytes** (`tobytes()`) instead of hashing **improves lookup times** in the visited states set.  
- This optimization prevents redundant state evaluations and reduces memory overhead.  

✅ **A* Outperforms Alternative Approaches**  
- **Dijkstra's algorithm** was impractical beyond **3×3** due to excessive state exploration.  
- **IDA\*** suffered from cycle repetition and was **less effective** than standard **A***.  
- **Parallel A\*** (JobLib) introduced overhead that negated potential speed gains.  

---

## 🔮 **Future Improvements**  

🔹 **Parallelization Strategies**  
- While the **thread-level parallel A\*** was not effective, experimenting with **multi-processing** or **GPU acceleration** could improve execution times.  

🔹 **More Advanced Heuristics**  
- Implementing **Pattern Databases** could further reduce the number of evaluated states.  
- Exploring **Machine Learning-based heuristics** to predict promising paths.  

🔹 **Adaptive Heuristic Scaling**  
- Instead of a fixed **multiplication factor**, a dynamic approach based on **search progress** could refine performance.  

---

## 📌 **Conclusion**  

This project successfully developed an **enhanced A\* algorithm** capable of solving **n×n sliding puzzles up to n=7** efficiently.  

By **combining multiple heuristics** and **optimizing state storage**, the approach finds **high-quality solutions** while keeping the number of evaluated states low.  

🔹 **For small grids (n ≤ 3)**: The algorithm guarantees the **optimal solution**.  
🔹 **For larger grids (n ≥ 4)**: The heuristic prioritizes **efficiency**, solving puzzles **that would otherwise be infeasible**.  

The balance between **solution quality and computational feasibility** makes this approach **scalable and practical for larger instances**. 🚀  

