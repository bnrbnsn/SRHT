import os
import numpy as np
import scipy.io as sio
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
import time
from optimal_precision_shrinkage import fopt
from helpers import mfhat, mfprimehat, theta1hat, theta2hat, myQ, QIS, compute_roc

# =============================================================================
# Preprocessing / Data Handling
# =============================================================================

def generate_synthetic_crawdad():
    """Generates synthetic data mirroring the structure of crawdad if real data is absent."""
    np.random.seed(42)
    p = 20
    T = 2000
    cov = np.diag(np.linspace(0.5, 3.0, p))
    A = np.random.randn(p, p)
    Q, _ = np.linalg.qr(A)
    cov = Q @ cov @ Q.T

    Z = np.random.multivariate_normal(np.zeros(p), cov, T).T
    motion = np.zeros(T, dtype=bool)

    # Inject motion events
    num_events = 15
    event_len = 30
    for _ in range(num_events):
        start = np.random.randint(0, T - event_len)
        motion[start:start+event_len] = True
        direction = np.random.randn(p, 1) * 2.5
        Z[:, start:start+event_len] += direction

    return Z, motion

def preprocess_data(data_file='dataLinear.mat', motion_file='motionCode.mat', save_mat_files=False):
    """
    Replicates preprocess.m behavior with high-precision arrays.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else os.getcwd()
    data_path = os.path.join(script_dir, data_file)
    motion_path = os.path.join(script_dir, motion_file)

    if not os.path.exists(data_path):
        print("!" * 60)
        print(f"WARNING: '{data_file}' not found at {data_path}!")
        print("Falling back to synthetic data. THIS WILL LOWER ROC PERFORMANCE!")
        print("!" * 60)
        return generate_synthetic_crawdad()
    else:
        print(f"Successfully loaded real data from: {data_file}")

    data = sio.loadmat(data_path)
    keys = [k for k in data.keys() if not k.startswith('_')]
    if 'dataLinear' in keys:
        X = data['dataLinear']
    else:
        X = data[keys[0]]
        
    row, col = X.shape
    Y = np.zeros_like(X, dtype=float)

    for ii in range(row):
        x = X[ii, :]
        dx = np.diff(x)
        idx0 = np.where(dx != 0)[0]
        index0 = np.concatenate(([0], idx0 + 1))
        if index0[-1] != col - 1:
            index0 = np.concatenate((index0, [col - 1]))
        value0 = x[index0]
        index1 = np.arange(col)
        f = CubicSpline(index0, value0, bc_type='not-a-knot')
        Y[ii, :] = f(index1)

    if save_mat_files:
        sio.savemat(os.path.join(script_dir, 'Y.mat'), {'Y': Y})

    P = 20
    trends = np.zeros_like(Y, dtype=float)
    for ii in range(row):
        for jj in range(P, col - P):
            trends[ii, jj] = np.mean(X[ii, jj - P : jj + P + 1])

    if save_mat_files:
        sio.savemat(os.path.join(script_dir, 'trends.mat'), {'trends': trends})

    Z = Y - trends
    Z = Z[:, P : col - P]
    
    alt_motion_path = os.path.join(script_dir, 'motion.mat')
    if not os.path.exists(motion_path) and os.path.exists(alt_motion_path):
        motion_path = alt_motion_path

    if os.path.exists(motion_path):
        mot_data = sio.loadmat(motion_path)
        keys = [k for k in mot_data.keys() if not k.startswith('_')]
        if 'motionCode' in keys:
            motion = mot_data['motionCode'].flatten()
        else:
            motion = mot_data[keys[0]].flatten()
        
        if len(motion) == col:
            motion = motion[P : col - P]
        elif len(motion) == col - 2*P:
            pass 
        print(f"Successfully loaded true motion labels from {motion_path}")
    else:
        print("Warning: Motion file missing, generating dummy labels")
        motion = np.zeros(Z.shape[1], dtype=bool)

    Z = Z[:, 24:]
    motion = motion[24:]
    motion = motion.astype(bool)

    if save_mat_files:
        sio.savemat(os.path.join(script_dir, 'Z.mat'), {'Z': Z})
        sio.savemat(os.path.join(script_dir, 'motion.mat'), {'motion': motion})

    return Z, motion

# =============================================================================
# Main Routine (Monte Carlo Loop)
# =============================================================================

def run_simulation(n, Z, motion, p, T, inactiveInds, numTrials=100):
    print(f"\n--- Running simulation for n={n} ---")
    
    all_resDL, all_resLR, all_resCWH = [], [], []
    all_resLW, all_resBai, all_resCQ, all_resHot = [], [], [], []
    
    for kk in range(numTrials):
        if (kk + 1) % max(1, (numTrials // 10)) == 0:
            print(f"  Trial {kk + 1} / {numTrials}")

        trainInds = np.random.choice(inactiveInds, n, replace=False)
        trainS = Z[:, trainInds]
        
        U, S_vals, _ = np.linalg.svd(trainS, full_matrices=False)
        lams = (S_vals**2) / n
        m = min(n, p)
        nzlams = lams[:m]
        full_lams = np.concatenate([nzlams, np.zeros(p - m)])

        mHatS = (p / n) * (trainS @ trainS.T)
        mHatSfrobSq = np.linalg.norm(mHatS, 'fro')**2
        rhoHatS = (p**2 + (1 - 2/p) * mHatSfrobSq) / ((p**2 - n*p - 2*n) + (n + 1 + 2*(n-1)/p) * mHatSfrobSq)
        
        chenShat = np.eye(p)
        for _ in range(50):
            chenShatOld = chenShat.copy()
            denomInv = np.linalg.solve(chenShat, trainS)
            denom = np.sum(trainS * denomInv, axis=0)
            chenStilde = (1 - rhoHatS) * (p / n) * (trainS @ np.diag(1 / denom) @ trainS.T) + rhoHatS * np.eye(p)
            chenShat = chenStilde * p / np.trace(chenStilde)
            if np.linalg.norm(chenShat - chenShatOld, 'fro') / p <= 1e-6:
                break

        QIS_eigs_ascending = QIS(p, n, np.flip(full_lams))
        lwlam = np.flip(QIS_eigs_ascending)
        
        inv_eigs_matched = fopt(p, n, nzlams, 'matched') 
        inv_eigs_iso = fopt(p, n, nzlams, 'isotropic') 
        
        xmin = np.sum(full_lams) / 100 / p
        xmax = 20 * np.max(full_lams)
        xs = np.logspace(np.log10(xmin), np.log10(xmax), 1000)
        Qs = myQ(xs, full_lams, p / n, False)
        myload0 = xs[np.nanargmax(Qs)]
        DL_inv_eigs = 1.0 / (full_lams + myload0)

        Z_U = U.T @ Z
        
        all_resDL.append( np.sum(Z_U * (DL_inv_eigs[:, None] * Z_U), axis=0) )
        all_resLR.append( np.sum(Z_U * (inv_eigs_iso[:, None] * Z_U), axis=0) )
        all_resBai.append( np.sum(Z_U * (inv_eigs_matched[:, None] * Z_U), axis=0) )
        all_resLW.append( np.sum(Z_U * ((1.0 / lwlam)[:, None] * Z_U), axis=0) )
        all_resCWH.append( np.sum(Z * np.linalg.solve(chenShat, Z), axis=0) )
        
        if n >= p:
            all_resHot.append( np.sum(Z_U * ((1.0 / full_lams)[:, None] * Z_U), axis=0) )
            
        x2avg = np.mean(trainS, axis=1)
        x1_sq = np.sum(Z**2, axis=0)
        x2_sq = np.sum(x2avg**2)
        x1_x2 = Z.T @ x2avg
        trainS_sq = np.sum(trainS**2)
        n2 = n 
        all_resCQ.append( 
            x1_sq + x2_sq * n2 / (n2 - 1) - 2 * x1_x2 - trainS_sq / (n2 * (n2 - 1)) 
        )

    motion_all = np.tile(motion, numTrials)
    
    rocs = {}
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
    # Set up LaTeX-style rendering matching EnergyDetectors06
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
    })

    np.random.seed(42)

    # =========================================================================
    # PART 1: CRAWDAD EMPIRICAL DATA PLOTS (Figure 6 a, b, c)
    # =========================================================================
    print("\n" + "="*50)
    print("Part 1: Generating Crawdad Empirical ROCs (Figure 6)")
    print("="*50)
    
    Z, motion = preprocess_data()
    assert motion.shape[0] == Z.shape[1], f"Alignment mismatch: motion {motion.shape[0]} vs Z {Z.shape[1]}"
    
    p, T = Z.shape
    inactiveInds = np.where(~motion)[0]
    
    train_sizes = [200, 250, 300]
    numTrials = 100 # formerly 1000***
    
    # Switch to the 1x3 structure matched from EnergyDetectors06
    # CHANGE THIS:
    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 4.5), squeeze=False)
    
    # TO THIS:
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5.7), squeeze=False)
    start_time = time.time()
    
    for idx, n in enumerate(train_sizes):
        rocs = run_simulation(n, Z, motion, p, T, inactiveInds, numTrials)
        ax = axes[0, idx]
        
        def add_curve(roc_data, linestyle, marker, color, lw, ms, mew, lbl, alpha=1, mfc='default'):
            if mfc == 'default':
                mfc = color
            
            # Proxy artist for the perfect legend entry (empty data, but labeled)
            ax.plot([], [], linestyle=linestyle, marker=marker, color=color, linewidth=lw, 
                    markersize=ms, markeredgewidth=mew, label=lbl, markerfacecolor=mfc, alpha=alpha) 

            # Lines and explicit spaced markers decoupled
            ax.plot(roc_data[0], roc_data[1], linestyle=linestyle, color=color, linewidth=lw, alpha=alpha)
            ax.plot(roc_data[2], roc_data[3], marker=marker, color=color, markersize=ms, 
                    markeredgewidth=mew, linestyle='None', alpha=alpha, markerfacecolor=mfc)

        # Build curves with exact styling from EnergyDetectors06
        # Build curves with exact styling from EnergyDetectors10
        ax.plot(rocs['DL'][0], rocs['DL'][0], 'k--', label='chance')
        add_curve(rocs['DL'], '--', 'v', '#0072BD', 2, 12, 1.5, 'LAPPW20', 0.8)
        add_curve(rocs['CWH'], '--', '^', '#D95319', 2, 12, 2, 'CWH11')
        
        add_curve(rocs['LW'], '--', '+', '#63D312', 3.25, 18, 3, 'LW22', 0.6) 

        add_curve(rocs['CQ'], '--', '.', 'red', 2, 16, 1.5, 'CQ10', 0.5)

        if 'Hot' in rocs:
            add_curve(rocs['Hot'], '--', '*', 'b', 2, 14, 1.5, 'Hotelling', 0.5)
            
        add_curve(rocs['Bai'], '-', 'o', 'goldenrod', 3.5, 16, 2.5, r'\textbf{SRHT}$\boldsymbol{(\Sigma)}$', 0.8, 'None') 
        add_curve(rocs['LR'], '-', 'x', 'm', 3.5, 16, 3, r'\textbf{SRHT(I)}', 0.8) 
        
        # Ledoit-Wolf (LW22) - called a second time with 'None' for proper legend deduplication matching
        add_curve(rocs['LW'], 'None', '+', '#63D312', 3.25, 18, 3, 'LW22', 0.6)
        # Apply specific EnergyDetectors06 log-axis formatting
        ax.set_xscale('log')
        ax.set_xlim([1e-3, 1])
        ax.set_xticks([1e-3, 1e-2, 1e-1, 1])
        ax.tick_params(axis='x', labelsize=18)
        ax.set_ylim([0, 1.05])
        ax.tick_params(axis='y', labelsize=18)
        
        ax.set_box_aspect(1) # <--- ADD THIS LINE HERE
        
        ax.set_title(r"$n=%d$" % n, fontsize=22)

    fig.tight_layout()
    
    # Legend Logic Extracted and Deduplicated
    handles, labels = axes[0,0].get_legend_handles_labels()
    by_label = {}
    for handle, label in zip(handles, labels):
        if label not in by_label:
            by_label[label] = handle

    leg = fig.legend(by_label.values(), by_label.keys(), loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0), fontsize=24)

    filename_main = 'Crawdad_Activity_Detection_Results.png' 
    fig.savefig(filename_main, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))
    print(f"Saved complete grid to: {filename_main}")

    #filename_main = '/Users/bnrbnsn/Downloads/latex/figures/Crawdad_Activity_Detection_Results.png' #!!*** change back to local path
    #fig.savefig(filename_main, dpi=300, bbox_inches='tight', bbox_extra_artists=(leg,))

if __name__ == "__main__":
    main()