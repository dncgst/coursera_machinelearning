COURSERA - Machine Learning - Week 4
====================================

## Neural Networks: Representation

Artificial neural networks were developed as simulating neurons or networks of neurons in the brain, following the **one learning algorith** hypothesis!

### Motivations

NN are learning algorithms, like linear regression and logistic regression, used to learn complex non-linear hypotheses (non-linear classification, )

### Neural Networks

In its simplest representation an artificial neuron has a Sigmoid or logistic _activation function_ $g(z)=\frac{1}{1+e^-z}$, where parameters $\theta$ are called _weights_.

In a neural networks, the _input_ layer is where we first input the features $x_0, x_1, x_2, \dots, x_n$. The middle _hidden_ layer is where the neural network $a_0, a_1, a_2, \dots, a_n$ is placed. There can be more than one hidden layer. The final _output_ layer is where a neuron output the computed value of $h_\Theta(x)$.

Notation:

- $a_i^{(j)}$: activation of unit $i$ in layer $j$
  - e.g. $a_1^{(2)}$ is the 1st unit of layer 2
      - $a_1^{(2)} =g(z) =g(\Theta_{10}^{(1)}x_0 + \Theta_{11}^{(1)}x_1 + \Theta_{12}^{(1)}x_2 + \Theta_{13}^{(1)}x_3)$, where input layer 1 $\in {x_1, x_2, x_3}$ and $x_0$ is the bias feature always = 1.
- $a_i^{(j)}=g(z_i^{(j)})$
- $\Theta^{(j)}$: matrix of weights controlling function mapping from layer $j$ to layer $j+1$
  - If network has $s_j$ units in layer $j$, $s_{j+1}$ in layer $j+1$, then $\Theta^{(j)}$ will be of dimension $s_{j+1}\times(s_j+1)$

Forward propagation - Vectorized implementation

With

$x$ or $a^{(1)}$ = column vector of $x_0, x_1, x_2, x_3, \dots, x_n$ at the 1st layer

and

$z^{(2)}$ = column vector of $z_1^{(2)}, z_2^{(2)}, z_3^{(2)}$ at the 2nd layer

At the 2nd layer we have:

$z^{(2)}=\Theta^{(1)}x$ or $z^{(2)}=\Theta^{(1)}a^{(1)}$

$a^{(2)}=g(z^{(2)})$

Add a bias activation function $a_0^{(2)}=1$ after you computed $a^{(2)}$!

At the 3rd layer we have:

$z^{(3)}=\Theta^{(2)}a^{(2)}$

$h_\Theta(x)=a^{(3)}=g(z^{(3)})$

To generalize: $h_\Theta(x)=a^{(j+1)}=g(z^{(j+1)})$

### Applications

- Complex non-linear classification
    - e.g., with binary $x_1, x_2$ we can implement AND, OR, NOT and XNOR logical functions
- Complex multiclass classification
  - e.g., one-vs-all method (see logistic regression)
