# ================================================================================
# Copyright (c) 2018 Benjamin Hou (bh1511@imperial.ac.uk)
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ================================================================================

import numpy as np
import torch

from geomstats.invariant_metric import InvariantMetric
from geomstats.special_euclidean_group import SpecialEuclideanGroup
from torch.autograd import Function


SE3_DIM = 6
N = 3

#weight = np.ones(SE3_DIM)
weight = np.array([1.,1.,1.,10.,10.,10.])
SE3_GROUP = SpecialEuclideanGroup(N)
metric = InvariantMetric( 
    group=SE3_GROUP, 
    inner_product_mat_at_identity=np.eye(SE3_DIM) * weight, 
    left_or_right='left')


class SE3GeodesicLoss(Function):
    """
    Geodesic Loss on the Special Euclidean Group SE(3), of 3D rotations
    and translations, computed as the square geodesic distance with respect
    to a left-invariant Riemannian metric.
    """

    @staticmethod
    def forward(ctx, inputs, targets):
        """
        PyTorch Custom Forward Function

        :param inputs:      Custom Function
        :param targets:     Function Inputs
        :return:
        """
        ctx.save_for_backward(inputs, targets)
        y_pred = inputs.data.cpu().numpy()
        y_true = targets.cpu().numpy()

        sq_geodesic_dist = metric.squared_dist(y_pred, y_true)
        batch_loss = np.sum(sq_geodesic_dist)

        return torch.FloatTensor([batch_loss])

    @staticmethod
    def backward(ctx, grad_output):
        """
        PyTorch Custom Backward Function

        :param grad_output: Gradients for equation prime
        :return:            gradient w.r.t. input
        """
        inputs, targets = ctx.saved_tensors
        y_pred = inputs.data.cpu().numpy()
        y_true = targets.cpu().numpy()
        
        tangent_vec = metric.log(
            base_point=y_pred,
            point=y_true)

        grad_point = - 2. * tangent_vec

        inner_prod_mat = metric.inner_product_matrix(
            base_point=y_pred)

        riemannian_grad = np.einsum('ijk,ik->ij', inner_prod_mat, grad_point)

        sqrt_weight = np.sqrt(weight)
        riemannian_grad = np.multiply(riemannian_grad, sqrt_weight)

        return grad_output.cuda() * torch.FloatTensor(riemannian_grad).cuda(), None
