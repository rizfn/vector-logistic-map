# A Vector Logistic Map

This project aims at extending the logistic map to `N` dimensions.

The resulting behaviour should allow for a way to interpolate between a coupled map lattice and a coupled map network.

We have the following structure:

$$ \vec{x}_\textrm{new} = \mathbb{A} \vec{x} (1-\vec{x}) $$

where $\vec{x}$ is an $N$ dimensional vector, and $\mathbb{A}$ is an $N\times N$ dimensional matrix.

### Link to CML

Consider the following structure of $\mathbb{A}$:

$$ \mathbb{A} = 
a\begin{pmatrix}
1-2\epsilon &\epsilon &  &  &  &  &  \\
\epsilon & 1-2\epsilon &\epsilon &  &  &  &  \\
& \epsilon & 1-2\epsilon & \epsilon &  &  &  \\
&  & \ddots & \ddots & \ddots \\
& & & & & \epsilon & 1-2\epsilon & \epsilon \\
& & & & & & \epsilon & 1-2\epsilon 
\end{pmatrix}$$

Where all the non-listed entries are 0. In other words, a tri-diagonal matrix, with $a-2 a\epsilon$ on the major diagonal, and $a\epsilon$ on the offset diagonals. Then, we can consider how the $i$ th element of $\vec{x}$ evolves in a single time step:

$$ 
\begin{align*}
x_i^\text{new} &= (a-2 a\epsilon) x_i(1-x_i) + a \epsilon (x_{i-1})(1-x_{i-1}) + a \epsilon (x_{i+1})(1-x_{i+1}) \\
&= \big(1-2\epsilon\big) a x_i(1-x_i) + \epsilon \bigg( a(x_{i-1})(1-x_{i-1}) + a (x_{i+1})(1-x_{i+1}) \bigg)  \\
\end{align*}
$$
Defining $f(x) = a(x)(1-x)$, we get
$$ 
\begin{align*}
x_i^\text{new} &= \big(1-2\epsilon\big) f(x_i) + \epsilon \big( f(x_{i-1}) + f(x_{i+1}) \big)  \\
\end{align*}
$$

Which is the equation of a standard CML in 1D.

### Link to GCM

Likewise, consider the following structure of $\mathbb{A}$:

$$ \mathbb{A} = 
a\begin{pmatrix}
1-\epsilon + \epsilon/N &\epsilon/N & \epsilon/N & \dots & \epsilon/N & \epsilon/N & \epsilon/N \\
\epsilon/N & 1-\epsilon + \epsilon/N & \epsilon/N & \dots & \epsilon/N & \epsilon/N & \epsilon/N \\
\vdots & \vdots & \ddots & \dots & \vdots & \vdots & \vdots \\
\epsilon/N & \epsilon/N & \epsilon/N & \dots & \epsilon/N & 1-\epsilon + \epsilon/N & \epsilon/N \\
\epsilon/N & \epsilon/N & \epsilon/N & \dots & \epsilon/N &\epsilon/N &  1-\epsilon + \epsilon/N \\
\end{pmatrix}$$

We can thus derive

$$ 
\begin{align*}
x_i^\text{new} &= \bigg(1-\epsilon + \frac\epsilon{N}\bigg) a (x_i)(1-x_i) + \sum_{j=0;\, j\neq i}^N \frac\epsilon{N} a x_j (1-x_j) \\
&= \big(1-\epsilon\big) a (x_i)(1-x_i) + \sum_{j=0}^N \frac\epsilon{N} a x_j (1-x_j) \\
&= \big(1-\epsilon\big) f(x_i) + \sum_{j=0}^N \frac\epsilon{N} f(x_j)  \\
\end{align*}
$$

Which is the equation of a GCM.
