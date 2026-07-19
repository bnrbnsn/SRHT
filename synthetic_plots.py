import numpy as np
import matplotlib.pyplot as plt
import math
import time
import sys
from optimal_precision_shrinkage import fopt
from helpers import mfhat, mfprimehat, theta1hat, theta2hat, myQ, QIS, compute_roc
        
def run_simulation(pow_idx, nu=math.inf, signal_is_isotropic=True, num_trials=100, p = 200, n = 300):
    """Runs Monte-Carlo trials for a given condition number power."""
    #p = 200 #1000
    #n = 100 # 500
    tmax = 10000 # 1000 * num_trials 
    if signal_is_isotropic:
        if nu == math.inf:
            mag = 9.75 # 10.5 # for sub-gaussian 
        else:
            mag = 13.75 # 14.5 # 12.5 # 16 # 10.5 # for heavy-tail
    else: # for anisotropic signals (Omega not equal Id)
        if nu == math.inf:
            mag = 9.75 # 12.5 # for (p, n) = (1000, 500)
        else:
            mag = 15 # 12.5 # 23.5
    nSpikes = 40

    print(f"\n--- Running simulation for kappa=10^{pow_idx}, nu={nu}, true_iso={signal_is_isotropic},  trials={num_trials} ---")
    
    # Generate population covariance eigenvalues
    part1 = 10.0 ** (pow_idx / nSpikes * (nSpikes + 1 - np.arange(1, nSpikes + 1)))
    #if pow_idx == 4: # ignore this conditional for now--original logic***
        #part2 = 10.0 ** (1 / (nSpikes * (p - nSpikes)) * (p - nSpikes + 1 - np.arange(1, p - nSpikes + 1)))
    if pow_idx == 0: # ignore this conditional for now--original logic***
        part2 = np.ones(p - nSpikes)
    else:
        part2 = 10.0 ** (1 / (nSpikes * (p - nSpikes - 1)) * (p - nSpikes - np.arange(1, p - nSpikes + 1)))
        
    #part2 = 10.0 ** (1 / (nSpikes * (p - nSpikes - 1)) * (p - nSpikes - np.arange(1, p - nSpikes + 1))) # more like paper definition***
    
    sigmavec = np.concatenate([part1, part2])
    sqrtSigma = np.diag(np.sqrt(sigmavec))
    # SigmaSq = np.diag(sigmavec ** 2) # for minimax***
    
    Z_iso = np.random.normal(size=(p, 2*p))
    V_iso, _, _ = np.linalg.svd(Z_iso)
    #print(V_iso.shape)
    #print(np.trace(V_iso.T @ V_iso))
    sqrtSigma = V_iso @ sqrtSigma
    
    # 5% of tests have the signal
    motion = np.zeros(tmax, dtype=bool)
    motion[int(0.95 * tmax):] = True

    # Pre-allocate score tracking
    all_resDL, all_resLR, all_resCWH = [], [], []
    all_resLW, all_resBai, all_resCQ, all_resHot = [], [], [], []

    for kk in range(num_trials):
        if (kk + 1) % max(1, (num_trials // 10)) == 0:
            print(f"  Trial {kk + 1} / {num_trials}")

        # Correctly condition the noise tail behavior for the reference data
        if nu == math.inf:
            trainS_white = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(p, n)) # sub-gaussian
        else:
            trainS_white = np.random.standard_t(nu, size=(p, n)) / np.sqrt(nu / (nu - 2)) # student-t
        
        trainS = sqrtSigma @ trainS_white
        
        # SVD and Eigenvalues
        U, S_vals, _ = np.linalg.svd(trainS, full_matrices=True)
        lams = (S_vals**2) / n
        m = min(n, p)
        nzlams = lams[:m]
        full_lams = np.concatenate([nzlams, np.zeros(p - m)])

        # Chen-Wiesel-Hero (CWH) supervised iteration (Fixing trace bug from original Matlab code)
        mHatS = (p / n) * (trainS @ trainS.T)
        mHatSfrobSq = np.linalg.norm(mHatS, 'fro')**2
        rhoHatS = (p**2 + (1 - 2/p) * mHatSfrobSq) / ((p**2 - n*p - 2*n) + (n + 1 + 2*(n-1)/p) * mHatSfrobSq)
        
        chenShat = np.eye(p)
        for _ in range(50):
            chenShatOld = chenShat.copy()
            denomInv = np.linalg.solve(chenShat, trainS)
            denom = np.sum(trainS * denomInv, axis=0)
            chenStilde = (1 - rhoHatS) * (p / n) * (trainS @ np.diag(1 / denom) @ trainS.T) + rhoHatS * np.eye(p)
            chenShat = chenStilde * p / np.trace(chenStilde)  # Trace normalized
            if np.linalg.norm(chenShat - chenShatOld, 'fro') / p <= 1e-6:
                break

        # Ledoit-Wolf quadratic inverse shrinkage
        QIS_eigs_ascending = QIS(p, n, np.flip(full_lams))
        lwlam = np.flip(QIS_eigs_ascending)
        
        
        # Proposed (Latimer-Robinson, isotropic version)
        # uOus = np.diag(U.T @ (SigmaSq @ U)) # minimax***
        uOus=np.ones(p) # isotropic***
        inv_eigs_iso = fopt(p, n, nzlams, 'isotropic') # isotropic signal prior
        
        # Diagonal loading optimization via grid search (vectorized)
        if n >= p:
            # Diagonal loading optimization via grid search (vectorized)
            xmin = (1 / p * np.sum(full_lams)) / 100
            xmax = 20 * np.max(full_lams)
            #xs = np.linspace(xmin, xmax, 10000)
            #Qs = myQ(xs, full_lams, p / n, signal_is_isotropic)
            #myload0 = xs[np.nanargmax(Qs)]

           
            # 1. Coarse grid search starting from xmax down to xmin
            # This avoids the non-smooth singularities inside the bulk of the eigenvalues
            xs = np.linspace(xmax, xmin, 1000)
            # xs = np.logspace(np.log10(xmax), np.log10(xmin), 1000)
            Qs = myQ(xs, full_lams, p / n, signal_is_isotropic)
            
            # Find the right-most local maximum (first peak encountered sliding down from xmax)
            best_idx = 0
            for i in range(1, len(Qs) - 1):
                if Qs[i] > Qs[i-1] and Qs[i] > Qs[i+1]:
                    best_idx = i
                    break
                    
            # Fallback if no local peak is found (monotonic behavior)
            if best_idx == 0 and Qs[-1] > Qs[0]:
                best_idx = np.nanargmax(Qs)
            
            # 2. Refine the maximum with a recursive grid search (bisection-like zoom)
            for _ in range(4):
                # Since xs is strictly descending, index - 1 is the upper bound
                local_upper = xs[max(0, best_idx - 1)]
                local_lower = xs[min(len(xs) - 1, best_idx + 1)]
                
                # Maintain descending order in the newly generated zoomed grid
                xs = np.linspace(local_upper, local_lower, 100)
                Qs = myQ(xs, full_lams, p / n, signal_is_isotropic)
                
                # Safe to use nanargmax now that we are isolated on the macro peak
                best_idx = np.nanargmax(Qs)
                
            myload0 = xs[best_idx]
            DL_inv_eigs = 1.0 / (full_lams + myload0)
            
        # Fully vectorized test data generation (10,000 samples at once)
        if signal_is_isotropic:
            meanDiffs = np.random.normal(size=(p,tmax))
            #meanDiffs = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(p, tmax)) 
        else:
            meanDiffs = sqrtSigma @ np.random.normal(size=(p,tmax))
        # myvec = meanDiffs[:,np.size(meanDiffs, 1)-1]
        # print(np.linalg.norm(myvec))
            
        if nu == math.inf:
            Z_white = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(p, tmax)) 
        else:
            Z_white = np.random.standard_t(nu, size=(p, tmax)) / np.sqrt(nu / (nu - 2)) 

        Z = sqrtSigma @ Z_white
        
        # Add isotropic, normalized signal to the selected H1 indices
        meanDiffs_signal = meanDiffs[:, motion]
        meanDiffs_norms = np.linalg.norm(meanDiffs_signal, axis=0)
        # Z[:, motion] += mag * (meanDiffs_signal / meanDiffs_norms)
        if signal_is_isotropic:
            Z[:, motion] += mag * meanDiffs_signal / np.sqrt(p)
        else:
            Z[:, motion] += mag * meanDiffs_signal / np.sqrt(p)

        # Vectorized Detector Scoring
        Z_U = U.T @ Z
        
        inv_eigs_matched = fopt(p, n, nzlams, 'matched') # covariance-matched prior
        
        inv_eigs_iso = inv_eigs_iso[:, None]
        inv_eigs_matched = inv_eigs_matched[:, None]        
            
        all_resLR.append( np.sum(Z_U * (inv_eigs_iso * Z_U), axis=0) )
        all_resLW.append( np.sum(Z_U * ((1.0 / lwlam)[:, None] * Z_U), axis=0) )
        all_resCWH.append( np.sum(Z * np.linalg.solve(chenShat, Z), axis=0) )
        # "Bai" Detector.  Actually covariance-matched proposed detector
        all_resBai.append( np.sum(Z_U * (inv_eigs_matched * Z_U), axis=0) )

        
        if n >= p:
            all_resDL.append( np.sum(Z_U * (DL_inv_eigs[:, None] * Z_U), axis=0) )
            all_resHot.append( np.sum(Z_U * ((1.0 / full_lams)[:, None] * Z_U), axis=0) )
            
        # Bai Detector
        #BaiN = n
        #BaiNum = np.linalg.norm(Z, axis=0)**2 - np.sum(full_lams)
        #Bn2 = (BaiN**2 / ((BaiN + 2)*(BaiN - 1))) * np.sum(full_lams**2) - (1 / BaiN) * abs(np.sum(full_lams)**2)
        #BaiDenom = np.sqrt(2 * (BaiN + 1) / BaiN * Bn2)
        #all_resBai.append( BaiNum / BaiDenom )
        
        # Chen Qin Detector
        x2avg = np.mean(trainS, axis=1)
        x1_sq = np.sum(Z**2, axis=0)
        x2_sq = np.sum(x2avg**2)
        x1_x2 = Z.T @ x2avg
        trainS_sq = np.sum(trainS**2)
        all_resCQ.append( x1_sq + x2_sq * n / (n - 1) - 2 * x1_x2 - trainS_sq / (n * (n - 1)) )

    # Flatten logic to process all trials efficiently
    motion_all = np.tile(motion, num_trials)
    
    rocs = {}
    if n >= p:
        rocs['DL'] = compute_roc(np.concatenate(all_resDL)[~motion_all], np.concatenate(all_resDL)[motion_all])
    rocs['LR'] = compute_roc(np.concatenate(all_resLR)[~motion_all], np.concatenate(all_resLR)[motion_all])
    rocs['LW'] = compute_roc(np.concatenate(all_resLW)[~motion_all], np.concatenate(all_resLW)[motion_all])
    rocs['CWH'] = compute_roc(np.concatenate(all_resCWH)[~motion_all], np.concatenate(all_resCWH)[motion_all])
    rocs['Bai'] = compute_roc(np.concatenate(all_resBai)[~motion_all], np.concatenate(all_resBai)[motion_all])
    rocs['CQ'] = compute_roc(np.concatenate(all_resCQ)[~motion_all], np.concatenate(all_resCQ)[motion_all])
    if n >= p:
        rocs['Hot'] = compute_roc(np.concatenate(all_resHot)[~motion_all], np.concatenate(all_resHot)[motion_all])

    return rocs
    
    

def main():
    # Set up LaTeX-style rendering in matplotlib without requiring local TeX installation
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}", # Load the math packages
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })
    
    np.random.seed(0)

    print("==================================================")
    print("      Robust Covariance Shrinkage Simulator       ")
    print("==================================================")
    print("1. Custom configuration (Specify tail behavior and signal prior)")
    print("2. Run 3x3 heavy-tail grid (isotropic case)")
    print("3. Run 3x3 heavy-tail grid (covariance-matched case)")
    mode = input("Select mode (1, 2, or 3): ").strip()
    
    print("\nEnter the dimension:")
    try:
        p_input = input("Dimension p (default 200): ").strip()
        p = int(p_input) if p_input else 200
    except ValueError:
        print("Invalid input, defaulting to dimension 200.")
        p = 200
        
    print("\nEnter number of training samples (default 300):")
    try:
        num_samples_input = input("Training set size n: ").strip()
        n = int(num_samples_input) if num_samples_input else 300
    except ValueError:
        print("Invalid input, defaulting to 300 samples.")
        n = 300
    
    print("\nEnter number of trials (default 10):")
    try:
        trials_input = input("Trials: ").strip()
        num_trials = int(trials_input) if trials_input else 10
    except ValueError:
        print("Invalid input, defaulting to 10 trials.")
        num_trials = 10
        
    if mode == '1':
        print("\nSelect TRUE signal prior:")
        print("1. Isotropic")
        print("2. Covariance-matched")
        true_iso = input("Choice (1 or 2): ").strip() == '1'
        
        print("\nSelect noise tail behavior:")
        print("1. Sub-Gaussian (nu = inf)")
        print("2. Student-t (nu = 10)")
        print("3. Student-t (nu = 8)")
        print("4. Student-t (nu = 6)")
        print("5. Student-t (nu = 4)")
        tail_choice = input("Choice (1-5): ").strip()
        
        nu_map = {'1': math.inf, '2': 10, '3': 8, '4': 6, '5':4}
        nu_val = nu_map.get(tail_choice, math.inf)
        
        run_configs = [(true_iso, [nu_val])]
        if true_iso:
             pow_configs = [0, 2, 4]
        else:
             pow_configs = [1, 2, 4]
    elif mode=='2':
        run_configs = [
            (True, [8, 6, 4])
        ]
        pow_configs = [0, 2, 4]
    else:
        run_configs = [
            (False, [8, 6, 4])
        ]
        pow_configs = [1, 2, 4]
    

        
    start_time = time.time()
    
    for (true_iso, nus_to_run) in run_configs:
        print(f"\n==================================================")
        print(f"Generating figure: TRUE Prior={'Iso' if true_iso else 'Cov'}")
        print(f"==================================================")
        
        # Main Grid Figure
        num_rows = len(nus_to_run)
        num_cols = len(pow_configs)
        # fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(20, 5 * num_rows), squeeze=False)
        # fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, 4.5 * num_rows), squeeze=False)
        # fig_top, axes_top = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 4.5), squeeze=False)
                # Calculate figure height dynamically to control aspect ratios separately.
        # We provide a baseline height per row, plus a fixed amount of overhead 
        # padding (1.5 inches) for the large column titles and the bottom legend.
        fig_height = 4.2 * num_rows + 1.5
        
        fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(15, fig_height), squeeze=False)
        fig_top, axes_top = plt.subplots(nrows=1, ncols=num_cols, figsize=(15, 5.7), squeeze=False)
 
        
        
        # Dedicated figure for the top row (Sub-Gaussian)
        #fig_top, axes_top = plt.subplots(nrows=1, ncols=num_cols, figsize=(20, 5), squeeze=False)
        
        for row, nu in enumerate(nus_to_run):
            for col, pow_idx in enumerate(pow_configs):
                rocs = run_simulation(pow_idx, nu=nu, signal_is_isotropic=true_iso, num_trials=num_trials, p=p, n=n)
                
                # Setup formatting strings for the title
                def plot_roc(ax, remove_cq=False):
                    # Helper function to plot lines and explicitly spaced markers
                    def add_curve(roc_data, linestyle, marker, color, lw, ms, mew, lbl, alpha=1,mfc='default'): #, mec='None'):
                        if mfc == 'default':
                            mfc = color
                        
                        if lbl == r'\textbf{SRHT(I)}$' or lbl == r'\textbf{SRHT}$\boldsymbol{(\Sigma)}$':
                            zorder=10
                            
                        # Proxy artist for the perfect legend entry (empty data, but labeled)
                        ax.plot([], [], linestyle=linestyle, marker=marker, color=color, linewidth=lw, 
            markersize=ms, markeredgewidth=mew, label=lbl, markerfacecolor=mfc, alpha=alpha) 
            # linestyle='None' if lbl=='LW22' else linestyle,

                        ax.plot(roc_data[0], roc_data[1], linestyle=linestyle, color=color, linewidth=lw, alpha=alpha)
                        
                        ax.plot(roc_data[2], roc_data[3], marker=marker, color=color, markersize=ms, markeredgewidth=mew, linestyle='None', alpha=alpha, markerfacecolor=mfc) 

                    # Thinner competitor lines
                    ax.plot(rocs['LR'][0], rocs['LR'][0], 'k--', label='chance')
                    if 'DL' in rocs:
                        add_curve(rocs['DL'], '--', 'v', '#0072BD', 2, 12, 1.5, 'LAPPW20', 0.8)
                    add_curve(rocs['CWH'], '--', '^', '#D95319', 2, 12, 2, 'CWH11')
                    
                    add_curve(rocs['LW'], '--', '+', '#63D312', 3.25, 18, 3, 'LW22', 0.6) #, 'black', 'black')
                    
                    # deprecated
                    ## add_curve(rocs['LW'], '--', 'None', '#63D312', 3.25, 18, 3, 'LW22') #, 'black', 'black')  # '#A2142F'
                    #
                    
                    #  CQ10
                    
                    add_curve(rocs['CQ'], '--', '.', 'red', 2, 16, 1.5, 'CQ10', 0.5) # '#63D312'
                    
                    if 'Hot' in rocs:
                        add_curve(rocs['Hot'], '--', '*', 'b', 2, 14, 1.5, 'Hotelling', 0.7)
                        
                    
                        
                    # Proposed Method 1: Thicker, bold color
                    if not true_iso or p > n or not nu == math.inf: #  suppress other method for Figure 4
                        add_curve(rocs['Bai'], '-', 'o', 'goldenrod', 3.5, 16, 2.5, r'\textbf{SRHT}$\boldsymbol{(\Sigma)}$', 0.8, 'None') 
                    
                    # Proposed Method 2: Thicker line
                    add_curve(rocs['LR'], '-', 'x', 'm', 3.5, 16, 3, r'\textbf{SRHT(I)}', 0.8) 
                    
                    # Ledoit-Wolf (LW22)
                    add_curve(rocs['LW'], 'None', '+', '#63D312', 3.25, 18, 3, 'LW22', 0.6) #, 'black')  # '#A2142F'
                    
                    




                    ax.set_xscale('log')
                    ax.set_xlim([1e-5, 1])
                    ax.set_xticks([1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1])
                    #if row == 0 and col == 2 and true_iso == True:
                        #ax.axvline(x=1e-4, color='gray', linestyle=':', linewidth=2)
                        
                    ax.tick_params(axis='x', labelsize=18) # Boosted for shrinking 
                    ax.set_ylim([0, 1.05])
                    ax.tick_params(axis='y', labelsize=18)                    
                    if row == 0:
                        # if num_rows == 0:
                            # xlabel_str = r"(a)" if col == 0 else r"(b)" if col == 1 else r"(c)"
                            # ax.set_xlabel(xlabel_str, fontsize=20)
                        #if pow_idx != 0:
                        title_str = r"$\kappa(\boldsymbol{\Sigma}) = 10^{%d}$" % pow_idx
                        #else:
                            #title_str = r"$\kappa(\boldsymbol{\Sigma}) = 1$"
                        ax.set_title(title_str, fontsize=28, y=1.025)
                        
                    #if row == 1:
                        # xlabel_str = r"(d)" if col == 0 else r"(e)" if col == 1 else r"(f)"                                                 
                        # ax.set_xlabel(xlabel_str, fontsize=20)
                        
                    # if row == 2:
                        # xlabel_str = r"(g)" if col ==0 else r"(h)" if col == 1 else r"(i)"
                        # ax.set_xlabel(xlabel_str, fontsize=20)
                        
                    # Add row labels exclusively to the leftmost column
                    if col == 0 and num_rows != 1:
                        print(nu)
                        row_lbl = r"$t_{%d}$" % nu
                        row_lbl = 'None' if nu==math.inf else row_lbl
                        #"sub-Gaussian data" if nu == math.inf else r"Student-$t$ data"                               
                        ax.set_ylabel(row_lbl, fontsize=34, labelpad=15, rotation=0, ha='right', va='center' )
                
                                    
                        

                    
                ax = axes[row, col]
                remove_cq = False
                if col == 0:
                    remove_cq = True
                plot_roc(ax, remove_cq)
                
                ax.set_box_aspect(1)
                
        # Explicitly reserve top 5% (for titles) and bottom 12% (for legend)
        fig.tight_layout() # (rect=[0, 0.12, 1, 0.95]) 
        handles, labels = axes[0,0].get_legend_handles_labels()

        # Safely deduplicate by keeping the *first* occurrence (the proxy artist)
        by_label = {}
        for handle, label in zip(handles, labels):
            if label not in by_label:
                by_label[label] = handle

        # Place the legend below the figure (negative Y) and assign it to a variable 'leg'
        leg = fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0), fontsize=24)





        mynu = 0 if nu==math.inf else nu
        #filename_main = f'/Users/bnrbnsn/Downloads/latex/figures/CovShrink_Grid_Iso{true_iso}_nu{mynu}_nt{num_trials}_n{n}.png' 
        #fig.savefig(filename_main, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))
        filename_main = f'CovShrink_Grid_Iso{true_iso}_nu{mynu}_nt{num_trials}_n{n}.png'
        fig.savefig(filename_main, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))

        print(f"Saved complete grid to: {filename_main}")
        plt.close(fig) # free memory
        
        # Save the dedicated top row figure
        #fig_top.tight_layout()
        #filename_top = f'CovShrink_TopRow_TrueIso{true_iso}_n{n}.png'
        #fig_top.savefig(filename_top, dpi=300, bbox_inches='tight')
        #print(f"Saved top row figure to: {filename_top}")
        #plt.close(fig_top)
    
    print(f"\nAll simulations completed in {time.time() - start_time:.1f} seconds.")

if __name__ == '__main__':
    main()