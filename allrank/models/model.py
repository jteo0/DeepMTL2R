import torch
import torch.nn as nn
from typing import List, Optional
from attr import asdict

from allrank.models.transformer import make_transformer
from allrank.utils.python_utils import instantiate_class


def first_arg_id(x, *y):
    return x


# Feature Gating: Dynamic Feature Gating (Soft Gating with Sigmoid)
class DynamicFeatureGate(nn.Module):
    """
    Soft Learnable Feature Gate for Dynamic Feature Gating (Feature Gating).

    Adds a learned sigmoid mask over the input feature dimension, allowing
    the model to suppress (but never fully zero-out) irrelevant or noisy
    features before they enter the main FCModel.

    Architecture: gate_weights (Linear) -> Sigmoid -> element-wise multiply with x

    Reference: GateNet (arXiv 2020), MaskNet (DNN for CTR 2021).
    """
    def __init__(self, n_features: int):
        """
        :param n_features: Number of input features (MSLR-WEB30K = 131).
        """
        super(DynamicFeatureGate, self).__init__()
        # A single linear layer that learns to produce a weight per feature
        self.gate_weights = nn.Linear(n_features, n_features)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the gate.
        :param x: input of shape [batch_size, slate_length, n_features]
        :return: gated input of same shape, with features soft-masked
        """
        # Produce a [0, 1] mask for each feature dimension
        gate_mask = self.sigmoid(self.gate_weights(x))
        return x * gate_mask

    def get_sparsity_ratio(self, threshold: float = 0.1) -> float:
        """
        Compute the fraction of input features effectively suppressed by the gate.
        A feature is considered 'gated off' if its average learned weight is below
        the threshold after applying sigmoid to the gate's bias (proxy for global mask).

        :param threshold: gate value below which a feature is considered inactive.
        :return: sparsity ratio in [0, 1].
        """
        with torch.no_grad():
            # Use the gate's bias term as a proxy for its "resting" gate value
            # (i.e., the gate value for a zero-input feature)
            bias_gate = self.sigmoid(self.gate_weights.bias)
            sparsity = (bias_gate < threshold).float().mean().item()
        return sparsity


# Matryoshka: Matryoshka Feature Projection (FCModel extension + output layer)
class MatryoshkaOutputLayer(nn.Module):
    """
    Multi-scale output layer for Matryoshka Representation Learning (Matryoshka).

    Produces independent scoring heads for each nesting dimension, enabling the
    model to learn a hierarchically rich representation where a smaller prefix
    of the embedding is itself meaningful.

    Adapted from MRL_Linear_Layer in the MRL reference repository (../MRL/MRL.py).
    """
    def __init__(self, d_model: int, nesting_dims: List[int], output_activation: Optional[str] = None):
        """
        :param d_model: total dimensionality of the encoder output (must be >= max(nesting_dims)).
        :param nesting_dims: list of embedding dimensions to supervise, e.g. [32, 64, 128, 256].
        :param output_activation: optional PyTorch activation applied before each scoring head.
        """
        super(MatryoshkaOutputLayer, self).__init__()
        assert d_model >= max(nesting_dims), (
            f"d_model ({d_model}) must be >= max nesting_dim ({max(nesting_dims)})"
        )
        self.nesting_dims = sorted(nesting_dims)
        self.d_model = d_model

        # One independent linear scoring head per nesting dimension
        # Each head maps from its specific nesting_dim -> 1 score
        for i, dim in enumerate(self.nesting_dims):
            setattr(self, f"nesting_head_{i}", nn.Linear(dim, 1))

        if output_activation is None:
            self.activation = nn.Identity()
        else:
            self.activation = instantiate_class("torch.nn.modules.activation", output_activation)

    def forward(self, x: torch.Tensor):
        """
        Forward pass: compute scores for each nesting dimension.
        :param x: encoder output of shape [batch_size, slate_length, d_model]
        :return: tuple of tensors, each of shape [batch_size, slate_length],
                 one per nesting dimension (from smallest to largest).
        """
        outputs = []
        for i, dim in enumerate(self.nesting_dims):
            # Slice the first `dim` dimensions of the embedding
            x_sliced = x[:, :, :dim]  # [batch_size, slate_length, dim]
            head = getattr(self, f"nesting_head_{i}")
            # Score: [batch_size, slate_length, 1] -> squeeze to [batch_size, slate_length]
            score = self.activation(head(x_sliced)).squeeze(dim=-1)
            outputs.append(score)
        return tuple(outputs)

    def score(self, x: torch.Tensor) -> torch.Tensor:
        """
        Score using the largest (most informative) nesting head only.
        Used for metric evaluation during training.
        :param x: encoder output of shape [batch_size, slate_length, d_model]
        :return: scores of shape [batch_size, slate_length]
        """
        # Use the last head (largest dimension)
        largest_dim = self.nesting_dims[-1]
        x_sliced = x[:, :, :largest_dim]
        last_head = getattr(self, f"nesting_head_{len(self.nesting_dims) - 1}")
        return self.activation(last_head(x_sliced)).squeeze(dim=-1)


# Original + Extended FCModel
class FCModel(nn.Module):
    """
    Fully connected neural network used as the input projection block of LTRModel.

    Extended to support Dynamic Feature Gating (Feature Gating) via the `use_gating` flag.
    When enabled, a DynamicFeatureGate is applied to the raw input features before
    the standard FC layers.
    """
    def __init__(self, sizes, input_norm, activation, dropout, n_features, use_gating: bool = False):
        """
        :param sizes: list of layer sizes (excluding input layer).
        :param input_norm: flag for LayerNorm on the input.
        :param activation: name of PyTorch activation function.
        :param dropout: dropout probability.
        :param n_features: number of input features.
        :param use_gating: if True, apply DynamicFeatureGate before FC layers (Feature Gating).
        """
        super(FCModel, self).__init__()
        sizes = list(sizes)
        sizes.insert(0, n_features)
        layers = [nn.Linear(size_in, size_out) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.input_norm = nn.LayerNorm(n_features) if input_norm else nn.Identity()
        self.activation = nn.Identity() if activation is None else instantiate_class(
            "torch.nn.modules.activation", activation)
        self.dropout = nn.Dropout(dropout or 0.0)
        self.output_size = sizes[-1]

        self.layers = nn.ModuleList(layers)

        # Feature Gating: Optional gating layer (applied before input_norm)
        self.use_gating = use_gating
        self.gating_layer = DynamicFeatureGate(n_features) if use_gating else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through FCModel.
        :param x: input of shape [batch_size, slate_length, n_features]
        :return: output of shape [batch_size, slate_length, output_size]
        """
        # Feature Gating: Apply soft gating mask first, before normalization
        if self.use_gating and self.gating_layer is not None:
            x = self.gating_layer(x)

        x = self.input_norm(x)
        for layer in self.layers:
            x = self.dropout(self.activation(layer(x)))
        return x


# Original OutputLayer (unchanged, used for Feature Gating and baseline)
class OutputLayer(nn.Module):
    """
    Standard output block reducing encoder output dimensionality to d_output=1.
    Used by the baseline and Dynamic Feature Gating.
    """
    def __init__(self, d_model, d_output, output_activation=None):
        super(OutputLayer, self).__init__()
        self.activation = nn.Identity() if output_activation is None else instantiate_class(
            "torch.nn.modules.activation", output_activation)
        self.d_output = d_output
        self.w_1 = nn.Linear(d_model, d_output)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.w_1(x).squeeze(dim=2))

    def score(self, x: torch.Tensor) -> torch.Tensor:
        if self.d_output > 1:
            return self.forward(x).sum(-1)
        else:
            return self.forward(x)


# LTRModel (extended to expose shared/task-specific parameters)
class LTRModel(nn.Module):
    """
    Full neural Learning to Rank model.
    Supports both standard OutputLayer (baseline/Feature Gating) and
    MatryoshkaOutputLayer (Matryoshka).
    """
    def __init__(self, input_layer, encoder, output_layer):
        super(LTRModel, self).__init__()
        self.input_layer = input_layer if input_layer else nn.Identity()
        self.encoder = encoder if encoder else first_arg_id
        self.output_layer = output_layer

    def prepare_for_output(self, x, mask, indices):
        return self.encoder(self.input_layer(x), mask, indices)

    def forward(self, x, mask, indices):
        return self.output_layer(self.prepare_for_output(x, mask, indices))

    def score(self, x, mask, indices):
        return self.output_layer.score(self.prepare_for_output(x, mask, indices))

    def shared_parameters(self):
        return list(self.input_layer.parameters()) + list(self.encoder.parameters())

    def task_specific_parameters(self):
        return list(self.output_layer.parameters())

    def last_shared_parameters(self):
        return []


# Factory function
def make_model(fc_model, transformer, post_model, n_features,
               use_mrl: bool = False,
               mrl_nesting_dims: Optional[List[int]] = None,
               use_gating: bool = False):
    """
    Factory function for instantiating LTRModel.

    :param fc_model: dict of FCModel kwargs (from config).
    :param transformer: TransformerConfig or None.
    :param post_model: dict of OutputLayer kwargs.
    :param n_features: number of input features.
    :param use_mrl: if True, use MatryoshkaOutputLayer instead of OutputLayer (Matryoshka).
    :param mrl_nesting_dims: nesting dimensions for Matryoshka, e.g. [32, 64, 128, 256].
    :param use_gating: if True, enable DynamicFeatureGate inside FCModel (Feature Gating).
    :return: LTRModel instance.
    """
    if fc_model:
        fc_model_kwargs = dict(fc_model)
        fc_model_kwargs['use_gating'] = use_gating
        fc_model_inst = FCModel(**fc_model_kwargs, n_features=n_features)
    else:
        fc_model_inst = None

    d_model = n_features if not fc_model_inst else fc_model_inst.output_size

    if transformer:
        transformer_inst = make_transformer(n_features=d_model, **asdict(transformer, recurse=False))
    else:
        transformer_inst = None

    # Choose output layer based on experiment mode
    if use_mrl:
        if mrl_nesting_dims is None:
            raise ValueError("mrl_nesting_dims must be provided when use_mrl=True.")
        output_activation = post_model.get("output_activation", None) if isinstance(post_model, dict) else post_model.output_activation
        output_layer = MatryoshkaOutputLayer(
            d_model=d_model,
            nesting_dims=mrl_nesting_dims,
            output_activation=output_activation,
        )
    else:
        post_model_kwargs = dict(post_model) if isinstance(post_model, dict) else asdict(post_model)
        output_layer = OutputLayer(d_model, **post_model_kwargs)

    model = LTRModel(fc_model_inst, transformer_inst, output_layer)

    # Initialize parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return model
