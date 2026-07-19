import numpy as np
import math

def mfhat(x, lams):
    """Marchenko-Pastur Stieltjes transform estimate."""
    # x: 1D array of shape (K,), lams: 1D array of shape (p,)
    return np.mean(1.0 / (lams[:, None] - x[None, :]), axis=0)

def mfprimehat(x, lams):
    """Derivative of the MP Stieltjes transform estimate."""
    return np.mean(1.0 / (lams[:, None] - x[None, :])**2, axis=0)

def theta1hat(x, lams, gamma):
    """Helper for LAPPW20 Diagonal Loading optimal search."""
    mf = mfhat(-x, lams)
    return (1.0 - x * mf) / (1.0 - gamma * (1.0 - x * mf))

def theta2hat(x, lams, gamma):
    """Helper for LAPPW20 Diagonal Loading optimal search."""
    mf = mfhat(-x, lams)
    mfp = mfprimehat(-x, lams)
    term1 = (1.0 - x * mf) / (1.0 - gamma * (1.0 - x * mf))**3
    term2 = x * (mf - x * mfp) / (1.0 - gamma * (1.0 - x * mf))**4
    return term1 - term2

def myQ(x, lams, gamma, is_isotropic):
    """Objective function to maximize for LAPPW20 regularization."""
    if is_isotropic:
        t1 = mfhat(-x, lams)
    else:
        t1 = theta1hat(-x, lams, gamma)
    t2 = theta2hat(-x, lams, gamma)
    return t1 / np.sqrt(t2)

def QIS(p, n, lams):
    """
    Ledoit-Wolf Quadratic Inverse Shrinkage (QIS) estimator.
    lams: sample eigenvalues sorted in ASCENDING order.
    Returns optimally shrunk eigenvalues in ascending order.
    """
    c = p / n
    h = min(c**2, 1/c**2)**0.35 / p**0.35
    invlambda = 1.0 / lams[max(0, p - n):p]
    
    # Broadcast differences (equivalent to Matlab's Lj - Lj')
    Lj_mat = np.broadcast_to(invlambda[None, :], (len(invlambda), len(invlambda)))
    Lj_i = invlambda[None, :] - invlambda[:, None]
    
    theta = np.mean(Lj_mat * Lj_i / (Lj_i**2 + h**2 * Lj_mat**2), axis=1)
    Htheta = np.mean(Lj_mat * (h * Lj_mat) / (Lj_i**2 + h**2 * Lj_mat**2), axis=1)
    Atheta2 = theta**2 + Htheta**2
    
    if p <= n:
        delta = 1.0 / ((1 - c)**2 * invlambda + 2 * c * (1 - c) * invlambda * theta + c**2 * invlambda * Atheta2)
    else:
        delta0 = 1.0 / ((c - 1) * np.mean(invlambda))
        delta = np.concatenate([np.repeat(delta0, p - n), 1.0 / (invlambda * Atheta2)])
        
    return delta
    
def compute_roc(scores_H0, scores_H1, num_pts=1000, num_markers=12):
    """Replicates Matlab's sliding threshold ROC computation."""
    # Fallback to prevent issues with empty arrays
    if len(scores_H0) == 0 or len(scores_H1) == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])
        
    labels = np.concatenate([np.zeros(len(scores_H0)), np.ones(len(scores_H1))])
    scores = np.concatenate([scores_H0, scores_H1])
    
    # Sort scores in descending order to compute exact ROC without binning artifacts
    sort_indices = np.argsort(scores)[::-1]
    labels_sorted = labels[sort_indices]
    
    # Cumulative sum gives exact true positives and false positives
    tpr = np.cumsum(labels_sorted) / max(1, len(scores_H1))
    fpr = np.cumsum(1 - labels_sorted) / max(1, len(scores_H0))
   
    # Generate perfectly uniform markers in log-space between 1e-5 and 1
    fpr_markers = np.geomspace(1e-5, 1.0, num_markers)
    tpr_markers = np.interp(fpr_markers, fpr, tpr)   
    
    # FIX: Downsample the curve using log-spaced FPR values instead of array indices!
    # This ensures the drawn line segments actually capture the log-scale details,
    # so the line perfectly intersects the interpolated markers.
    if len(fpr) > num_pts:
        fpr_curve = np.geomspace(1e-6, 1.0, num_pts)
        tpr = np.interp(fpr_curve, fpr, tpr)
        fpr = fpr_curve
    
    # Prepend (0,0) to anchor the origin robustly
    fpr = np.insert(fpr, 0, 0.0)
    tpr = np.insert(tpr, 0, 0.0)
    
    return fpr, tpr, fpr_markers, tpr_markers

