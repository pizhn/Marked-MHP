# Marked-MHP

A library for the simulation and estimation of **Marked Multivariate Hawkes Process**(Marked-MHP).

Code referred and modified from this [Soni Sandeep's excellent implementation](https://github.com/sandeepsoni/MHP)!

> MIT License
> 
> Copyright (c) 2017 Sandeep Soni

1. Can now estimate and simulate **marks** in MHP.
2. Streamlined the optimization process – it's now vectorized.
3. No need for a full n*n parameter; it supports edges.
4. You can optimize by individual node instead of the whole network.
Overall, these changes make the estimation much faster!

Points 2, 3 and 4 significantly speeds up the estimation process.

I used the MLE (maximum likelihood estimation) for parameter estimation. If MLE isn't your preference, you might want to check out [Tick](https://github.com/X-DataInitiative/tick). But ensure Tick's Hawkes Process aligns with your needs.

*Stay tuned: I'll be dropping more details, comments, and examples soon!
