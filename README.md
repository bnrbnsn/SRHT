README

Code to enable reproduction of figures from "Spectrally Robust Covariance Shrinakge of Hotelling's T^2 in High Dimensions" by B. Robinsoin and V. Latimer, as well as additional testing of the algorithm if desired.  (https://arxiv.org/pdf/2502.02006)

To generate Figure 6, run the following command from the current directory:

python3 crawdad_plots.py

To generate the remaining Figures after Figure 3, run the following command, which will bring up a menu:

python3 synthetic_plots.py 

Option 1 allows selection of dimension p, sample size n, number of Monte Carlo trials (choose 200, 300, and 3000, respectively, for main-body figures; choose 200, 100, and 3000, respectively, for supplemental material figures), and tail shape (1 for sub-Gaussian (nu=0), or one of several possible Student-t shapes t_nu, for nu = 10, 8, 6, or 4).  Option 2 produces a 3x3 grid of plots displaying behavior in the heavy-tailed data regime, for an isotropic signal prior.  Similarly, option 3 produces a 3x3 grid object plots for the heavy-tailed regime when the signal prior is covariance-matched (in particular, anisotropic).  Again, 3000 Monte Carlo trails are used in the paper.

Figures 1-3 do not pertain to the paper's algorithm's performance and are not presently included, but may be in a future update.
