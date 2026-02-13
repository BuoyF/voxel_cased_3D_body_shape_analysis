"""
Evaluation metrics for body composition prediction
"""

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr


def compute_metrics(targets, preds, comp_names):
    """
    Compute comprehensive metrics for all 66 composition targets
    
    Args:
        targets: Array of shape [N, 66] (ground truth)
        preds: Array of shape [N, 66] (predictions)
        comp_names: List of 66 target names
    
    Returns:
        Dictionary with metrics for each target
    """
    results = {}
    
    for i, name in enumerate(comp_names):
        y_true = targets[:, i]
        y_pred = preds[:, i]
        
        # Skip if no variation in targets
        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            results[name] = {
                'RMSE': np.nan,
                'MAE': np.nan,
                'R2': np.nan,
                'PearsonR': np.nan,
            }
            continue
        
        results[name] = {
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'PearsonR': pearsonr(y_true, y_pred)[0],
        }
    
    return results
