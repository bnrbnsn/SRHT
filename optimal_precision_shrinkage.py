def optimal_precision_shrinkage(p, n, nz_lams, prior_type, hbar=None, xi=None):
    r"""
    $\color{codepurple}\textbf{\text{Description}}$: Proposed nonlinear precision shrinkage estimator, generalized to the $\color{codegreen}p > n$ regime but also valid for $\color{codegreen}n < p$.  
    $\color{codepurple}\textbf{\text{Output}}$: (p x 1) array $\color{codegreen}f=\color{codepurple}f^o(\lambda_i)$ (f_opt) that approximately optimizes
    $\color{codegreen}\mathrm{tr}(f(\bS)\myOmega)^2/\tr(\bfr f(\bS) \myXi f(\bS))$ (appropriately scaled) in the limit as $\color{codegreen}n,p\to\infty$ and $\color{codegreen}p/n\to\phi$, where $\color{codegreen}\lambda_i$ are the eigenvalues of the sample covariance matrix $\color{codegreen}\bS$, in descending order.
    $\color{codepurple}\textbf{\text{Inputs}}$: 
    - Dimension p, sample size n
    - nz_lams: descending nonzero sample eigenvalues.
    - prior_type (signal prior): 'isotropic', 'matched' (covariance-matched), or determined by hbar if any other argument is given
    - hbar[i]: estimated signal-prior spectral parameters of $\color{codegreen}\myOmega$: $\color{codegreen} \overline{h}(\lambda_i) = d\omega_\infty/d\mu_\infty(\lambda_i)$
    - xi[i]: estimated spectral parameters of $\color{codegreen}\myXi$: $\color{codegreen}\xi_\infty(\lambda_i)$, defined analogously for the matrix $\color{codegreen}\myXi$ to how $\color{codegreen}\overline{h}(\lambda_i)$ is defined for $\color{codegreen}\myOmega$
    """
    
    m = len(nz_lams)
    phi = p / n
    hn = p**(-1/3) # Bandwidth for kernel estimation

    # Semi-circular kernel and its Hilbert transform
    def k_func(x):
        return 1 / (2 * np.pi) * np.sqrt(np.maximum(4 - x**2, 0))
    def Hk_func(x):
        return (-x + np.sign(x) * np.sqrt(np.maximum(x**2 - 4, 0))) / (2 * np.pi)

    # Matrices for taking quick convolutions with k(x) and Hk(x)
    lam_diffs = nz_lams[:, None] - nz_lams[None, :]
    arg_array = lam_diffs / (nz_lams[None, :] * hn)
    Hk_arr = Hk_func(arg_array)
    k_arr = k_func(arg_array)

    # Estimate density $\color{codegreen}w(x)$ and its Hilbert transform $\color{codegreen}\mathcal{H}w(x)$ at each $\color{codegreen}\lambda_i \ne 0$
    Hw = 1 / (p * hn) * np.sum(Hk_arr / nz_lams[None, :], axis=1)
    w = 1 / (p * hn) * np.sum(k_arr / nz_lams[None, :], axis=1)

    # Construct $\color{codegreen}g(x)$ and denominator norm squared $\color{codegreen}b(x)^2 + B(x)^2$
    g_func = np.maximum(1 - phi, 0) - phi * np.pi * nz_lams * Hw
    norm_sq = g_func**2 + (phi * np.pi * nz_lams * w)**2

    # Nonlinear shrinkage function d, or $\color{codegreen}\tilde{d}_n(x)$, of (Ledoit-Wolf, 2020)
    d_nz = nz_lams / norm_sq # Non-degenerate sample eigenvalues d_nz
    Hw_zero = (1 - np.sqrt(max(1 - 4 * hn**2, 0))) / (2 * np.pi * n * hn**2) * np.sum(1.0 / nz_lams)
    d_zero = n / (np.pi * max(p - n, 1) * Hw_zero) # Degenerate ei-val
    d = np.concatenate([d_nz, np.repeat(d_zero, max(0, p - m))])
        
    # Assign xi and hbar (based on prior_type string)
    if prior_type == 'matched':
        hbar = d
    elif prior_type == 'isotropic':
        hbar = np.ones(p)
    elif hbar is None:
        exit("Error: Invalid prior type or no value given for hbar")
    if xi is None: # Default: In present setting $\color{codegreen}\myXi = \bfr$
        xi = d_nz # $\color{codegreen}\xi(x) =\tilde{d}_n(x)$

    # Estimate Hilbert transform of $\color{codegreen}h_c$
    H_hc = 1 / (p * hn) * np.sum(Hk_arr * (hbar[:m] / nz_lams)[None, :], axis=1)
    Gbar = - phi * np.pi * nz_lams

    # Base partial solution $\color{codegreen}f_*$ from (S.7)
    f_sub_star = (g_func * hbar[:m] + Gbar * H_hc) / (nz_lams * xi[:m])
    H_lambda_fsubstar = 1 / (p * hn) * np.sum(Hk_arr * f_sub_star[None, :], axis=1)

    # Non-singular solution $\color{codepurple}f^*$ from main body/(S.7) 
    f_opt = g_func * f_sub_star + phi * np.pi * H_lambda_fsubstar
    # Note: 2nd term equivalent to -H_w[Gbar*f_sub_star] by def of Gbar

    # Corrections for the singular regime ($\color{codegreen}\phi > 1$)
    if phi > 1:
        h_s = 1/p * np.sum(hbar[n:p]) # Weight for null eigenvalues
        H_w_over_xi = 1 / (p * hn) * np.sum(Hk_arr / (xi * nz_lams)[None, :], axis=1)
        
        # Adjustment to obtain $\color{codegreen}\tilde{f}^o_c$, as in $\eqref{eq:fc}$: Note that $\color{codegreen}g(x)/x$ equals $\color{codegreen}-\phi \pi \mathcal{H}w(x)$ on $\color{codegreen}F$
        f_opt += phi * h_s * (g_func / nz_lams / xi + phi * np.pi * H_w_over_xi)
        
        # Equation $\eqref{eq:fs}$ implementation: For scalar component $\color{codegreen}\tilde{f}^o_s(0)$, use empirical averages np.mean() and np.sum()
        f_s = phi**2 * h_s * 1/p * np.sum(1 / (xi * nz_lams)) + 1 / n * np.sum(f_sub_star)
        f_opt = np.concatenate([f_opt, np.repeat(f_s, p - m)]) 
        
    # Return $\color{codegreen}\max\{f^o, 0\}$ to ensure positivity
    return np.maximum(f_opt, 0)
