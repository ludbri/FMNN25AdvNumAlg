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
- normal wall, u(x)=15    (only outer walls)
- heated wall, u(x)=40
- wall with windows, u(x)=5

# Modelling
The temperature at the point $x$ is $u(x)$.
$x$ is modelled using matrix coordinates, starting in the top-left with x_1 vertical, x_2 horizontal.

To find the steady state distribution, the laplace equation is (approximately) solved:
$$\Delta u(x) = 0, x\in\Omega\subset \mathbb{R}^2$$

To solve it, it is first discretized on a grid with spacing $\Delta x$.

The discrete approximation on the grid is denoted $u_{i,j}\approx u(x_{i,j})$. Then, the 2nd order differences are:
$$\Delta u(x_{i,j}) \approx \frac{u_{i,j+1}+u_{i,j-1}-4u_{i,j}+u_{i+1,j}+u_{i-1,j}}{\Delta x^2}$$,
where $\Delta x = h > 0$ is the distance between discretization points along each dimension.


# Solution approach:
For each room and iteration, the heat values are found by solving $Au=b$. $A$ is a $|u|$x$|u|$ matrix and $u$ and $b$ are column vectors.

Start by using Dirichlet (heat) conditions to solve room 2.
The solution gives Neumann (heat derivative) conditions for use in rooms 1&3.
Solve 1&3.
Update $u^k$ with exponential smoothing, and keep iterating.

## Condition types
For each discretized point $u_{i,j}$, there is a corresponding condition in $A$ and $b$:
- If $x_{i,j}\in\Gamma_h$, then $u_{i,j}=T_h$
- If $x_{i,j}\in\Gamma_{wh}$, then $u_{i,j}=T_{wh}$
- If $x_{i,j}\in\Gamma_w$, then $u_{i,j}=T_w$
- If $x_{i,j}\in\Gamma_1\setminus(\Gamma_1\cap\Gamma_{\delta\Omega})$ and room 1, then [incoming Neumann 1]
- If $x_{i,j}\in\Gamma_1\setminus(\Gamma_1\cap\Gamma_{\delta\Omega})$ and room 2, then $u_{i,j}=u_{DC1}(j)$
- If $x_{i,j}\in\Gamma_2\setminus(\Gamma_2\cap\Gamma_{\delta\Omega})$ and room 3, then [incoming Neumann 2]
- If $x_{i,j}\in\Gamma_2\setminus(\Gamma_2\cap\Gamma_{\delta\Omega})$ and room 2, then $u_{i,j}=u_{DC2}(j)$
- If $x_{i,j}$ is an interior point, then $\Delta u_{i,j}=0$

## Assumptions and observations:
- For $k=0$, it is assumed that the [Dir] and [Neu] conditions are equal to 0.
- It is assumed that boundary points (corners) that are in both the outer boundary $\delta\Omega$ and $\Gamma_1$ or $\Gamma_2$ are only in $\delta\Omega$.
- The western (eastern) corners in room 1 (3) have no influence on the solution as their neighbours are always other walls with fixed temperature. Thus, their condition is irrelevant.
The same is true for the NW and SE corners of room 2.
- The NE (SW) corder of room 1 (3) is always a standard wall no matter which room is evaluated. 
- The SE (NW) corner of room 1 (3) is treated according to the room currently considerd: when room 1 (3) is evaluated, it is a standard wall with $T_w$, when room 2 is evaluated, it is a $T_{wh}$ ($T_h$) wall. This has no impact on the result, as the neighours are both with fixed temperature when room 2 is evaluated and it will ultimately only matter when it is a standard wall.

## Computing the outgoing boundary conditions from each domain:
- The outgoing Dirichlet condition (DC) from rooms 1 and 3 is simply the resulting value of $u_{i,j}$ in the boundary.
- The outgoing Neumann conditions (NC) from room 2 are computed

From the discretized Lagrange equation, the outgoing Dirichlet condition (here in direction (0,1)) is:
$$\frac{u_{i,j+1}+u_{i,j-1}-4u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h^2} = 0$$
$$\iff \frac{u_{i,j+1}+u_{i,j-1}-4u_{i,j}+u_{i+1,j}+u_{i-1,j}}{\Delta x^2} = -\frac{u_{DC}(i)}{h^2}$$
$$\iff u_{i,j+h}-4u_{i,j}+u_{i-h,j}+u_{i,j-h} = -u_{DC}(i)$$

From the discretized Lagrange equation, the outgoing Neumann condition (here in direction (0,1)), $u_{NC}(i)=\frac{u_{i,j+1}-u_{i,j}}{h}$ is:
$$\frac{u_{i,j+1}+u_{i,j-1}-4u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h^2} = 0$$
$$\iff \frac{u_{NC}(i)}{h}+\frac{u_{i,j-1}-3u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h^2} = 0$$
$$\iff \frac{u_{i,j-1}-3u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h^2} = - \frac{u_{NC}(i)}{h}$$
$$\iff \frac{u_{i,j-1}-3u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h} = - u_{NC}(i)$$


## Incoming boundary conditions
For the incoming Dirichlet conditions, the receiving sub-domain includes the constraint $u_{i,j}=u_DC(i)$ for $x_{i,j}\in\Gamma_n$.

For the incoming Neumann conditions, the receiving sub-domain includes the constraint (here in direction $(0,1)$) $u_{i,j+1}-u_{i,j}=u_{NC}(i)$.




## Boundary condition exploration example
So the question on NC is which side to include, consider the reduced 1-d example of points

$\Omega = \{0, 1, 2, 3, 4\}$
Let $\Omega_1 = \{0,1,2\}$, $\Omega_2 = \{2,3,4\}$, $\Gamma_1=2$

Then, the Dirichlet condition is simply $u_3$.
On evaluation of $\Omega_1$, $u_2$ is free to vary.
On evaluation of $\Omega_2$, $u_3$ is fixed to the DC value.
On $\Omega 1$, it will be solved from the condition related to point $x_{i-1,j}$. - $u_{DC}(i)$ is simply recovered from the solution $u$ as if it was an orderinary point.

The Neumann condition is from $\Omega_2$ to $\Omega_1$. The question is, is it the derivative from $3\rightarrow 2$ or from $2\rightarrow 1$?
According to lecture notes, it is the derivative from the boundary point outwards, i.e., $2\rightarrow 1$!
So, on $\Omega_2$, it is recovered from the solution $u_{\Omega_2}^k$ as $\\$
(1d case) $\frac{u_{3}-u_{2}}{h} = - u_{NC}$ $\\$
(2d case, direction $(0,1)$) $\frac{u_{i,j-1}-3u_{i,j}+u_{i+1,j}+u_{i-1,j}}{h} = - u_{NC}(i)$.
So, it can be evaluated efficiently as a matrix operation with a precomputed matrix.



# Error bound of discretization
...