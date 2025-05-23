# Adaptive Gradient Learning for Spiking Neural Networks by Exploiting Membrane Potential Dynamics (IJCAI, 2025)

**This is the official Pytorch implementation**

**Abstract**: Recent advancements have focused on directly training high-performance spiking neural networks (SNNs) by estimating the approximate gradients of spiking activity through a continuous function with constant sharpness, known as surrogate gradient (SG) learning. However, as spikes propagate within neurons and among layers, the distribution of membrane potential dynamics (MPD) will deviate from the gradient-available interval of fixed SG, hindering SNNs from searching the optimal solution space. To maintain the stability of gradient flows, SG needs to align with evolving MPD. Here, we propose a novel adaptive gradient learning for SNNs by exploiting MPD, namely MPD-AGL. It fully accounts for the underlying factors contributing to membrane potential shifts and establishes a dynamic association between SG and MPD at different timesteps to relax gradient estimation, which provides a new degree of freedom for SG learning. Experimental results demonstrate that our method achieves excellent performance at low latency. Moreover, it increases the proportion of neurons that fall into the gradient-available interval compared to fixed SG, effectively mitigating the gradient vanishing problem.

**Updating.....**
