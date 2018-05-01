#!/usr/bin/env python3
# Copyright 2017 Christian Henning
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
@title           :util/lin_alg.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :03/30/2017
@version         :1.0
@python_version  :3.5.2

This module contains several helper functions implementing basic linear algebra
algorithms to modify tensors resp. to certify specific criteria.
"""

import numpy as np
from sklearn.preprocessing import normalize
import scipy

def gram_schmidt(matrix):
    """Orthogonalize a matrix using Gram-Schmidt algorithm.

    Args:
        matrix: A NxM array. If M>N, then the last M-N columns will be filled
            with zeros.
    Returns:
        A matrix containing orthogonal columns.
    """
    # Note, we could also compute gram-schmidt in place.
    ret_mat = np.zeros(matrix.shape)

    # Project y onto x
    proj = lambda x,y : np.dot(x,y) / np.dot(x,x) * x

    for i in range(0,min(matrix.shape)):
        ret_mat[:,i] = matrix[:,i]
        for j in range(i):
            ret_mat[:,i] -= proj(ret_mat[:,j], matrix[:,i])

    return ret_mat

def check_orthogonality(mat, norm=True):
    """Check the orthogonality of a given matrix and return the mean and std.
        dev. of mutual column vector dot products.

    Args:
        mat: An NxM matrix.
        norm: Whether the matrix has to be normalized. This can be disabled, if
            the given matrix is already normalized. Note, if the matrix is not
            normalized, the returns are no interpretable resp. comparable
            orthogonality measure.

    Returns:
        mean: The mean dot product of two distinct column vectors.
        std_dev: The standard deviation of the dot product of two distinct
            column vectors.
    """
    if norm:
        nmat = normalize(mat, norm='l2', axis=0, copy=True)
    else:
        nmat = mat

    dps = []
    for i in range(nmat.shape[1]):
        for j in range(i+1, nmat.shape[1]):
            dps.append(np.dot(nmat[:,i], nmat[:,j]))

    return np.mean(dps), np.std(dps)

def check_euclidean_distances(mat, norm=True):
    """Check the distribution of euclidean distances of column vectors in a
        given matrix and return the mean and std. dev.

    Args:
        mat: An NxM matrix.
        norm: Whether the matrix has to be normalized. This can be disabled, if
            the given matrix is already normalized. Note, if the matrix is not
            normalized, the returns are no interpretable resp. comparable
            measures.

    Returns:
        mean: The mean euc. distance of two distinct column vectors.
        std_dev: The standard deviation of the euc. distance of two distinct
            column vectors.
    """
    if norm:
        nmat = normalize(mat, norm='l2', axis=0, copy=True)
    else:
        nmat = mat

    eds = []
    for i in range(nmat.shape[1]):
        for j in range(i+1, nmat.shape[1]):
            eds.append(np.linalg.norm(nmat[:,i] - nmat[:,j]))

    return np.mean(eds), np.std(eds)

def evaluate_sparsity(mat, eps=1e-15):
    """Evaluate the sparsity of a given matrix. Therefore, the mean and std.
    dev. of the percentage of zeros per column are computed.

    Args:
        mat: An NxM matrix.
        eps: Threshold, that determines which values are considered as zero.

    Returns:
        mean: The mean of the relative percentage of the number of zeros per
            column.
        std_dev: The standard deviation of the relative percentage of zeros per
            column.
    """
    nzs = []
    for c in range(mat.shape[1]):
        nz = 0

        for r in range(mat.shape[0]):
            if mat[r,c] < eps:
                nz += 1

        nzs.append(nz/mat.shape[0])

    return np.mean(nzs), np.std(nzs)



def extend_orthogonal_base(mat):
    """Extend an NxM matrix with M>N, that already has N orthogonal vectors,
    such that the remaining M-N vectors have a minimal dot product to all
    others as much as possible.

    The method implements the following approach (this is just a guess to be a
    good solution, no properties have been proven, as this should be a rare
    case to be happening).

    x_1 to x_N are an orthogonal basis of the R^N. We do now extend the basis
    by the vectors x_N+1 to x_M as follows:

        x_j = 1/(j-1) SUM_i=0^j-1 x_i for j in [N+1,M]

    Example:
        x_N+1 = 1/N SUM_i=0^N x_i

        Hence: <x_i, x_N+1> = 1/N ||x_i||

        x_N+2 = 1/(N+1) SUM_i=0^N+1 x_i

        Hence:
            <x_i, x_N+2> = 1/(N+1) ||x_i|| + 1/(N+1)*1/N ||x_i||
            for i in [0,N]

            <x_N+1,x_N+2> = 1/(N+1)*1/N SUM_i=0^N ||x_i|| +
                            1/(N+1) ||x_N+1||

    Args:
        mat: An NxM matrix with M>N, where the first N columns are orthogonal.

    Returns:
        Returns the given matrix, where the last M-N columns have been computed
        given the above scheme.
    """
    n, m = mat.shape
    ret_mat = mat.copy()

    for j in range(n,m):
        ret_mat[:,j] = np.zeros(n)

        for i in range(j-1):
            ret_mat[:,j] += ret_mat[:,i]

        ret_mat[:,j] /= j

    return ret_mat

def introduce_sparcity(mat, threshold, percentage=False):
    """Introduce sparcity to a dense matrix, meaning, setting all values below
    a certain threshold to zero or set a given percentage of smallest entries
    to zero.

    Args:
        mat: The matrix, that shall be converted to a sparse matrix.
        threshold: A number, that either represents a threshold value or a
            relative percentage (range between 0 and 1).
        percentage: Whether the parameter 'threshold' shall be interpreted as
            threshold or percentage.

    Returns:
        The given matrix with introduced sparcity.
    """
    n, m = mat.shape
    ret_mat = mat.copy()

    if percentage:
        sorted_indices = np.argsort(ret_mat, axis=0)
        num_zeros = round(threshold*n)
        for c in range(m):
            for r in range(num_zeros):
                ret_mat[sorted_indices[r,c],c] = 0
    else:
        for c in range(m):
            for r in range(n):
                if ret_mat[r,c] <= threshold:
                    ret_mat[r,c] = 0

    return ret_mat

def compute_null_space(mat, eps=1e-15):
    """Compute the left null space of a matrix. I.e. a basis of the orthogonal
    complement of the span of the columns from the given matrix.

    The method has been copied and modified from:
    http://stackoverflow.com/questions/5889142/python-numpy-scipy-finding-the-null-space-of-a-matrix
    License: https://creativecommons.org/licenses/by-sa/3.0/

    Args:
        mat: A square matrix.
        eps: Threshold, that decides if an entry is considered as zero.
    Returns:
        A basis of the null space for the transposed input matrix.
    """
    u, s, vh = scipy.linalg.svd(mat.T)
    null_mask = (s <= eps)
    null_space = scipy.compress(null_mask, vh, axis=0)
    return null_space.T

if __name__ == '__main__':
    pass


