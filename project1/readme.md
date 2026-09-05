# Project 1 - solving heat distribution

This project is my attempt at solving project 1:
To find the resulting steady state temperature distribution in a 3-room appartment while making use of parallel computation and MPI.

# Problem
The rooms are of size 1x1, 1x2, 1x1 with layout:

```
    +---+---+
    |   | 3 |
+---+ 2 +---+
| 1 |   |
+---+---+
```

The between-room borders are denoted:
- 1|2 is G1
- 2|3 is G2

There are heaters on:
- the west wall of 1
- the north wall of 2
- the east wall of 3

The south wall of 2 has a big window

The wall temperatures are:
- normal wall, u(x)=15    (assumed only outer walls and not G1, G2?)
- heated wall, u(x)=40
- wall with windows, u(x)=5



The temperature at the point $x$ is $u(x)$.
$x$ is modelled using matrix coordinates, starting in the top-left with x_1 vertical, x_2 horizontal.

To find the steady state distribution, the laplace equation is (approximately) solved:
$$\Delta u(x) = 0, x\in\Omega\subset \mathbb{R}^2$$

To solve it, it is first discretized on a grid with spacing $\Delta x$.

The discrete approximation on the grid is denoted $u_{i,j}\approx u(x_{i,j})$. Then, the 2nd order differences are:
$$\Delta u(x_{i,j}) \approx \frac{u_{i,j+1}+u_{i,j-1}-4u_{i,j}+u_{i+1,j}+u_{i-1,j}}{\Delta x^2}.$$



Idea: start by using Dirichlet (heat) conditions to solve room 2.
The solution gives Neumann (heat derivative) conditions for use in rooms 1&3.
Solve 1&3.
Update uk with exponential smoothing, and keep iterating.