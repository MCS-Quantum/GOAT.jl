# Reference frames

Within this package we also provide the support to define dynamical reference frames. A choice of reference frame (sometimes called an "rotating frame" or an "interaction picture") is a mathematical and physical utility used to simplify quantum dynamics. In general a reference frame is provided by a time-dependent unitary operator $V(t)$. If the unitary is time-independent then the reference frame coincides with a change of basis, thus one can think of a dynamical reference frame as a time-dependent choice of basis functions. 

With a time-dependent unitary operator $V(t)$ defined, one derives a new effective Hamiltonian within the reference frame using the systems Hamiltonian $H(t)$ and the Schrodinger equation:

```math
\begin{align*}
    i\hbar \partial_t \bigg[V(t)\ket{\psi(t)}\bigg] &= i\hbar \bigg[ (\partial_t V(t))\ket{\psi(t)} + V(t) \partial_t\ket{\psi(t)} \bigg] \\
    &= i\hbar \bigg[ (\partial_t V(t)) + V(t) \frac{1}{i \hbar} H(t) \bigg] \ket{\psi(t)} \\
    &= i\hbar \bigg[ (\partial_t V(t)) V^\dagger(t) + \frac{1}{i \hbar} V(t)H(t)V^\dagger(t) \bigg] V(t)\ket{\psi(t)} \\
    &= H_{eff}(t) \bigg[ V(t) \ket{\psi(t)} \bigg].
\end{align*}
```

While the reference frame is defined as a unitary operator, it is often more compact to specify the reference frame by specifying the generator of the unitary. In other words specify a Hermitian operator $A$ such that $V(t) = \exp[-\frac{it}{\hbar} A]$. This operator may be derived from the system's Hamiltonian and is often chosen in such a way that the effective quantum dynamics within the reference frame is simplified. The most common example taught in contemporary quantum mechanics is known as the [interaction picture](https://en.wikipedia.org/wiki/Interaction_picture). 

However, the interaction picture is only a particular choice of refence frame! Many other references frames can be defined which are esspecially interesting or useful (See for example, the original [DRAG paper](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.103.110501) where a time-dependent reference frame is used to derive some of the most effective and common control pulses used today.)

**Reference frames are particularly important in experimental quantum devices!** One example of this is occurs in assessing the "idling" error. Idling error occurs when coherent quantum dynamics occur even when the system is not being controlled, i.e. there is a component of the Hamiltonian that is non-zero at every time. This leads to decoherence and errors. A reference frame can be chosen to mininimize these unwanted errors. Thus, all controls need to be chosen such that they generate the desired evolutions within the specified reference frame. 

For example, in experimental systems local clocks are used to "lock" a reference frame and synchronize classical electronics. If there is noise in the clock this appears as noise within the quantum dynamics and can affect quantum device performance (See [this paper](https://www.nature.com/articles/npjqi201633)). 

GOAT.jl permits a definition of the generator $A$ and will efficiently compute the quantum dynamics in the rotating frame. How to specify this operator in your simulations is described further in the documentation and the examples.