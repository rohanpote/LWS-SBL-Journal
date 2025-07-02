# Theory and Practice of Light-Weight Sequential SBL Algorithm: An Alternative to OMP

The code implements the algorithms proposed in the following paper, and also generates the plots presented in the same:

R. R. Pote and B. D. Rao, "Theory and Practice of Light-Weight Sequential SBL Algorithm: An Alternative to OMP"

Please refer the above paper if you find the proposed ideas or code relevant for your work.

# Description of the work:

We propose a low complexity forward selection algorithm for the sparse signal recovery (SSR) problem based on the sparse Bayesian learning (SBL) formulation. The proposed algorithm, called as light-weight sequential SBL (LWS-SBL), offers an alternative to the widely used iterative and greedy algorithm known as orthogonal matching pursuit (OMP). In contrast to OMP, which models the unknown sparse vector as a deterministic variable, the same is modeled as a stochastic variable within LWS-SBL. Specifically, the proposed algorithm is derived from the stochastic maximum likelihood estimation framework, and it iteratively selects columns that maximally increase the likelihood. We derive efficient recursive procedure to update the internal parameters of the algorithm, and maintain a similar asymptotic computational complexity as OMP. Additional two perspectives, one based on array processing beamforming interpretations and the other based on a local high-resolution analysis, are provided to understand the underlying differences in the mechanisms of the two algorithms. They reveal avenues where LWS-SBL improves over OMP. These are verified in the numerical section in terms of improved support recovery performance. Similar to the counterparts in OMP, for SSR problems involving parametric dictionaries, the flexibility of the proposed approach is demonstrated by extending LWS-SBL to recover multi-dimensional parameters, and in a gridless manner.
