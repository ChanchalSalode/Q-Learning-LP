import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import random

# -------------------------------
# Define the LP problem (for comparison)
# -------------------------------
c = [-10, -8]  # Coefficients for objective function (maximize 10x + 8y, so negate for minimization)
A = [[1, 2], [1, 3], [5, 2]]  # Constraint coefficients
b = [10, 12, 18]  # Right-hand side values

# Solve using Simplex
res = opt.linprog(c, A_ub=A, b_ub=b, method='highs')
print("Simplex Method:")
print("Optimal Solution:", res.x)
print("Optimal Value:", -res.fun)  # Negate because linprog minimizes

# -------------------------------
# Q-learning Parameters
# -------------------------------
epsilon = 1  # Exploration rate
alpha = 0.9    # Learning rate
gamma = 0.9    # Discount factor

grid_size = 0.3  # Step size in exploration

# Q-table to store the estimated value for each (x,y) state.
Q = {}
exploration_path = []

# A helper function to discretize the state (rounding to one decimal place)
def get_state(x, y):
    return (round(x, 1), round(y, 1))

# A helper function to check feasibility of (x,y)
def is_feasible(x, y):
    return all([
        x >= 0,
        y >= 0,
        x + 2 * y <= 10,
        x + 3 * y <= 12,
        5 * x + 2 * y <= 18
    ])

# -------------------------------
# Q-learning Exploration
# -------------------------------
# We run episodes with a fixed number of steps (1000).
# The reward is given only at the final (terminal) step; all intermediate rewards are zero.
for episode in range(500):
    x, y = 0, 0  # Start at (0,0)
    for step in range(1000):
        state = get_state(x, y)
        # Define possible actions: move in x or y direction.
        actions = [(grid_size, 0), (-grid_size, 0), (0, grid_size), (0, -grid_size)]
        
        # Epsilon-greedy action selection.
        if random.uniform(0, 1) < epsilon:
            action = random.choice(actions)
        else:
            q_values = [Q.get(get_state(x + dx, y + dy), 0) for dx, dy in actions]
            action = actions[np.argmax(q_values)]
        
        new_x, new_y = x + action[0], y + action[1]
        
        # Check if the new state is feasible.
        if not is_feasible(new_x, new_y):
            continue  # Skip actions that lead out of the feasible region.
        
        # Save the state for visualization.
        exploration_path.append((new_x, new_y))
        
        # Only at the terminal step, give the reward as the objective function value.
        if step == 999:  # Last step in the episode
            reward = 10 * new_x + 8 * new_y
        else:
            reward = 0.0
        
        # Update Q-table using the Q-learning rule.
        new_state = get_state(new_x, new_y)
        Q[state] = (1 - alpha) * Q.get(state, 0) + alpha * (reward + gamma * Q.get(new_state, 0))
        
        # Move to the new state.
        x, y = new_x, new_y

# Find the state with the highest Q-value.
optimal_state = max(Q, key=Q.get)
optimal_value = Q[optimal_state]

print("\nQ-learning Method:")
print("Optimal Solution:", optimal_state)
print("Approximate Optimal Value:", optimal_value)

# -------------------------------
# Plot Feasible Region and Solutions
# -------------------------------
x_vals = np.linspace(0, 6, 100)
y1 = (10 - x_vals) / 2
y2 = (12 - x_vals) / 3
y3 = (18 - 5 * x_vals) / 2
y_final = np.minimum(np.minimum(y1, y2), y3)
y_final[y_final < 0] = 0

plt.figure(figsize=(6, 6))
plt.fill_between(x_vals, 0, y_final, color='lightblue', alpha=0.5)
plt.scatter(*optimal_state, color='red', label='Q-learning Solution')
plt.scatter(res.x[0], res.x[1], color='green', label='Simplex Solution')

# Plot the exploration path (the visited states).
if exploration_path:
    exploration_x, exploration_y = zip(*exploration_path)
    plt.plot(exploration_x, exploration_y, 'k-', alpha=0.3, label='Q-learning Path')

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("LP Feasible Region and Solutions")
plt.show()