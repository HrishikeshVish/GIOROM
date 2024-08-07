import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLPLinear
from .segment_csr import segment_csr

class IntegralTransform(nn.Module):
    """Integral Kernel Transform (GNO)
    Computes one of the following:
        (a) \int_{A(x)} k(x, y) dy
        (b) \int_{A(x)} k(x, y) * f(y) dy
        (c) \int_{A(x)} k(x, y, f(y)) dy
        (d) \int_{A(x)} k(x, y, f(y)) * f(y) dy

    x : Points for which the output is defined
    y : Points for which the input is defined
    A(x) : A subset of all points y (depending on
           each x) over which to integrate
    k : A kernel parametrized as a MLP
    f : Input function to integrate against given
        on the points y

    If f is not given, a transform of type (a)
    is computed. Otherwise transforms (b), (c),
    or (d) are computed. The sets A(x) are specified
    as a graph in CRS format.

    Parameters
    ----------
    mlp : torch.nn.Module, default None
        MLP parametrizing the kernel k. Input dimension
        should be dim x + dim y or dim x + dim y + dim f
    mlp_layers : list, default None
        List of layers sizes speficing a MLP which
        parametrizes the kernel k. The MLP will be
        instansiated by the MLPLinear class
    mlp_non_linearity : callable, default torch.nn.functional.gelu
        Non-linear function used to be used by the
        MLPLinear class. Only used if mlp_layers is
        given and mlp is None
    transform_type : str, default 'linear'
        Which integral transform to compute. The mapping is:
        'linear_kernelonly' -> (a)
        'linear' -> (b)
        'nonlinear_kernelonly' -> (c)
        'nonlinear' -> (d)
        If the input f is not given then (a) is computed
        by default independently of this parameter.
    """

    def __init__(
        self,
        mlp=None,
        mlp_layers=None,
        mlp_non_linearity=F.gelu,
        transform_type="linear",
    ):

        super().__init__()

        assert mlp is not None or mlp_layers is not None

        self.transform_type = transform_type

        if (
            self.transform_type != "linear_kernelonly"
            and self.transform_type != "linear"
            and self.transform_type != "nonlinear_kernelonly"
            and self.transform_type != "nonlinear"
        ):

            raise ValueError(
                f"Got transform_type={transform_type} but expected one of "
                "[linear_kernelonly, linear, nonlinear_kernelonly, nonlinear]"
            )

        if mlp is None:
            #print("MLP LAYERS = ", mlp_layers)
            self.mlp = MLPLinear(layers=mlp_layers, non_linearity=mlp_non_linearity)
        else:
            self.mlp = mlp

    """"
    

    Assumes x=y if not specified
    Integral is taken w.r.t. the neighbors
    If no weights are given, a Monte-Carlo approximation is made
    NOTE: For transforms of type 0 or 2, out channels must be
    the same as the channels of f
    """

    def forward(
        self,
        y,
        neighbors,
        x=None,
        f_y=None,
        weights=None,
    ):
        """Compute a kernel integral transform

        Parameters
        ----------
        y : torch.Tensor of shape [n, d1]
            n points of dimension d1 specifying
            the space to integrate over.
        neighbors : dict
            The sets A(x) given in CRS format. The
            dict must contain the keys "neighbors_index"
            and "neighbors_row_splits." For descriptions
            of the two, see NeighborSearch.
        x : torch.Tensor of shape [m, d2], default None
            m points of dimension d2 over which the
            output function is defined. If None,
            x = y.
        f_y : torch.Tensor of shape [n, d3], default None
            Function to integrate the kernel against defined
            on the points y. The kernel is assumed diagonal
            hence its output shape must be d3 for the transforms
            (b) or (d). If None, (a) is computed.
        weights : torch.Tensor of shape [n,], default None
            Weights for each point y proprtional to the
            volume around f(y) being integrated. For example,
            suppose d1=1 and let y_1 < y_2 < ... < y_{n+1}
            be some points. Then, for a Riemann sum,
            the weights are y_{j+1} - y_j. If None,
            1/|A(x)| is used.

        Output
        ----------
        out_features : torch.Tensor of shape [m, d4]
            Output function given on the points x.
            d4 is the output size of the kernel k.
        """

        if x is None:
            x = y

        rep_features = y[neighbors["neighbors_index"]]
        if f_y is not None:
            in_features = f_y[neighbors["neighbors_index"]]
        #print("rep_feats Nans = ", torch.isnan(rep_features.view(-1)).sum().item())
        #print("in_feats Nans = ", torch.isnan(in_features.view(-1)).sum().item())
        #print("INSIDE INTEGRAL TRANSFORM")
        #print("Y  SHAPE = ", y.shape)
        #print("X Shape = ", x.shape)
        #print("F_Y shape = ", f_y.shape)
        num_reps = (
            neighbors["neighbors_row_splits"][1:]
            - neighbors["neighbors_row_splits"][:-1]
        )
        #print("NUM REPS = ", num_reps.shape)
        self_features = torch.repeat_interleave(x.cuda(), num_reps, dim=0)
        #print("self_features Nans = ", torch.isnan(self_features.view(-1)).sum().item())
        #print("SELF FEATURES SHAPE = ", self_features.shape)
        agg_features = torch.cat([rep_features, self_features], dim=1)
        agg_features = rep_features
        #print("AGG features Nans = ", torch.isnan(agg_features.view(-1)).sum().item())
        #print("AGG Features shape = ", agg_features.shape)
        #print("IN FEATURES = ", in_features.shape)
        if f_y is not None and (
            self.transform_type == "nonlinear_kernelonly"
            or self.transform_type == "nonlinear"
        ):
            if(len(agg_features.shape) == len(in_features.shape)):
                agg_features = torch.cat([agg_features, in_features], dim=1)
            else:
                agg_features = torch.unsqueeze(agg_features, dim=2)
                agg_features = torch.cat([agg_features, in_features],dim=2)
                agg_features = torch.reshape(agg_features, shape=(agg_features.shape[0]*agg_features.shape[1], 5))
        #print("AGG Features shape = ", agg_features.shape)
        
        rep_features = self.mlp(agg_features)
        #print("REP Features shape = ", rep_features.shape)
        #print("rep features after MLP Nans = ", torch.isnan(rep_features.view(-1)).sum().item())
        if f_y is not None and self.transform_type != "nonlinear_kernelonly":
            rep_features = rep_features * in_features
        #print("REP FEATURES AFTER PRODUCT = ", rep_features.shape)
        if weights is not None:
            rep_features = weights[neighbors["neighbors_index"]] * rep_features
            reduction = "sum"
        else:
            reduction = "mean"
        #print("rep features after bunch of ops Nans = ", torch.isnan(rep_features.view(-1)).sum().item())
        #print("REP Before segment = ", rep_features.shape)
        #print("NEIGHBORS = ", neighbors["neighbors_row_splits"])
        out_features = segment_csr(
            rep_features, neighbors["neighbors_row_splits"], reduce=reduction
        )

        #print("OUT FEATURES After segment = ", out_features.shape)
        #print("out features after MLP Nans = ", torch.isnan(out_features.view(-1)).sum().item())
        return out_features
