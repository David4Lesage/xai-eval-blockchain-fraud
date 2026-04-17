"""XAI explainer wrappers with a uniform interface.

Every explainer exposes a ``.explain(X, indices)`` method that returns an
attribution matrix of shape ``(len(indices), n_features)``. This lets the
evaluation modules treat SHAP, LIME, GNNExplainer, etc. interchangeably.
"""

from xai_blockchain_framework.xai.lime_wrapper import LimeTabularExplainer
from xai_blockchain_framework.xai.shap_wrapper import ShapTreeExplainer

__all__ = [
    "ShapTreeExplainer",
    "LimeTabularExplainer",
]
