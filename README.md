# Marked-MHP

A library for simulation and estimation of **Marked Multivariate Hawkes Process**(Marked-MHP).

Code referred and modified from this [Soni Sandeep's excellent implementation](https://github.com/sandeepsoni/MHP)!

Did following updates:
1. Supports estimation and simulation of **mark** in MHP
2. Vectorized optimization process for estimation
3. Supports edges (network don't need to be n*n full parameter)
4. Supports optimization by node, instead of full network

Points 1, 3 and 4 significantly speeds up the estimation process.

(Details, comments and examples will be added in the near future)
