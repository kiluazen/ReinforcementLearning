## Code Structure
Folder 1: $\epsilon$-greedy
This has the implementations of Thompson Sampling, UCB, KL_UCB algorithms implemented 

Folder 2: $\pi$ (policy) iteration
- planner.py : has implementions of Value Iteration, Howard Policy Iteration, Linear Programming to find optimal policy.

- encoder.py : Encodes the football game given inside [football_problem](./football_problem.html) to a format that *planne.py* can take.

- decoder.py : Results from planner.py are represented in the states form given in the problem statement.

## Implementaions

Please go through the code alongside the mathematical algorithms, it's commented well.

I will update this file with thorough walk through of the code.

## Reference

Lecture Slides of [CS 747 IITB ](https://www.cse.iitb.ac.in/~shivaram/teaching/cs747-a2023/index.html)

## Acknowlegements

To Prof. [Shivaram Kalyanakrishnan](https://www.cse.iitb.ac.in/~shivaram/), For desiging one of the best courses at IITBombay, The pace of the course and his delivery is awesome. I am so glad I took it.

## License
This project is licensed under the [MIT License](LICENSE).

