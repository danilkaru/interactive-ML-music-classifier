# Helper file containing function for calculating embedding space variance
# Is used in update_plot function in app_layout.py.

import numpy as np


def compute_MAEST_embedding_space_variance(embeddings):
    """
    Function which calculates embedding space variance.
    NOTE: assumes embedding is either a np.array, or a MAEST embedding
    """
    embedding_variances = []
    for e in embeddings:
        if isinstance(e,np.ndarray):
            # Process as embedding which is stored as np.array
            embedding_result = e
        else:
            # Process as MAEST embedding 
            embedding_result = e.mean(dim=0).detach().numpy()        
        
        embedding_variances.append(np.var(embedding_result))

    return embedding_variances



