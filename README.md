# Q-Learning for Solving Linear Programming Problems

This project demonstrates how **Q-learning**, a reinforcement learning (RL) technique, can be used to approximately solve a **Linear Programming (LP)** problem. The goal is to explore how learning-based algorithms can tackle classical optimization problems and compare their performance with exact methods like the **Simplex algorithm**.

---

## 📌 Problem Statement

We consider the following LP problem:

**Maximize**

10x + 8y

**Subject to:**

x + 2y ≤ 10
x + 3y ≤ 12
5x + 2y ≤ 18
x, y ≥ 0

---

## 🚀 Approach

- The LP is solved using two methods:
  1. **Simplex Method** via `scipy.optimize.linprog`
  2. **Q-Learning**: An RL-based method that explores the feasible space and learns the optimal point through trial and error

- The feasible region is visualized, along with:
  - The exploration path of the Q-learning agent
  - Final solutions from both Q-learning and Simplex methods

---

## 📊 Visualization

<p align="center">
  <img src="images/feasible_region_plot.png" alt="Feasible Region Plot" width="400">
</p>

---

## 🛠️ Files Included

Q-Learning-LP/
├── q_learning_lp.py           # Main Python script
├── README.md                  # This file
├── requirements.txt           # List of dependencies
└── images/
└── feasible_region_plot.png  # Optional: plot image (save after running the script)

---

## 📦 Installation

Clone the repository and install required libraries:

```bash
git clone https://github.com/your-username/Q-Learning-LP.git
cd Q-Learning-LP
pip install -r requirements.txt



⸻

▶️ Running the Code

python q_learning_lp.py

After execution:
	•	Console prints both solutions (Simplex and Q-learning)
	•	A plot window displays the feasible region, exploration path, and solutions

⸻

📚 Learnings and Extensions
	•	Shows how RL can approximate solutions in constrained environments
	•	Can be extended to:
	•	Deep Q-Learning (for higher dimensions)
	•	Non-linear or mixed-integer problems
	•	Alternative RL algorithms (e.g., SARSA, PPO)

⸻

🧑‍💻 Author

Chanchal Kumar Salode
PhD Research Scholar, IIT Delhi
GitHub: @ChanchalSalode

⸻
