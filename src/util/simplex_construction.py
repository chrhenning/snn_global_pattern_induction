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
@title           :util/simplex_construction.py
@author          :ch
@contact         :christian@ini.ethz.ch
@created         :04/05/2017
@version         :1.0
@python_version  :3.5.2

This module provides methods to construct an n-dimensional simplex, that has
mutually equidistant vertices. The simplex can always be transformed, such that
all its vertices lie on a the surface of a unit sphere.

Note, it can easily be proven, that in an n-dimensional at most n+1 mutually
equidistant points can exist. Therefore, this module only allows to construct
at most n+1 equidistant points in an n dimensional space. In future, this could
be extended by an non-optimal algorithm that solves the following optimization
problem. Sample m (m > n+1) distinct points on the onit sphere. Minimize there
mutual distance while maximizing the convex hull of the point cloud (points
should only be moved along the surface of the sphere).

The gain in euclidian distance that can be achieved by this method (in contrast
to orthogonal patterns) decreases with increasing n.
Proof Sketch:
    n: pattern dimensionality
    m: number of classes/patterns

    Assumption: m <= n

    We are starting with a fully orthonormal matrix p'. The pattern p' defines
    a mutually equidistant simplex in an m-1 dimensional space. For simplicity,
    we assume p' is the identity matrix (if not, it can always be rotated to
    become it). To maximize the distance between vertices, one has to move and
    scale the simplex such that it fits a unit sphere (cmp. method
    _move_simplex_in_unit_sphere).

    Let c denote the centroid of p'.
    (FIXME I mixed m and n for the rest of the proof.)
    c = [1/n, ..., 1/n]
    ||c|| = sqrt(n*(1/n^2)) = sqrt(1/n)

    Let r be the radius of the sphere we are considering (in our case r = 1).
    Let the connecting line between the origin and c be x. If we move along x,
    the vector r (pointing to an orthogonal axis) shrinks.
    To be more precise, moving a vector r' orthogonal to x that connects the x
    axes and the sphere surface, r' its magnitude shrinks the farther away it
    gets from the origin.
    ||r'|| = r*sin(acos(||c||/r))
           = r*sin(acos(1/(r*sqrt(n))))

    Hence, moving the centroid by the amount ||c|| to the origin (while
    increasing its size to get radius r), will result in a size gain:
        dr = r/||r'|| = r/(r*sin(acos(1/(r*sqrt(n)))))

    However, this only works if the vertices of the simplex are on the surface
    of the sphere with radius r. Therefore, we only consider r==1 and neglect r
    from now on.

    One interesting thing to see now is, that ||r'|| is the same as the
    distance from a vertex of p' to c (the radius of the sphere that encloses
    p'). Let a be an arbitrary reference point. We choose
        a = [1,0,...0]
        => d = a - c = [ 1-1/n, -1/n, ..., -1/n ]
        => ||d|| = sqrt((1-1/n)^2 + (n-1)*(-1/n)^2)
                 = sqrt(1 - 1/n)
    Thus ||r'|| = sin(acos(1/sqrt(n))) = sqrt(1 - 1/n) = ||d||.

    The total gain is 1 - ||d||.
"""

import configuration as config
from util.config_exception import ConfigException
from util import lin_alg

import numpy as np

import logging
logger = logging.getLogger(config.logging_name)

def _get_mutequi_point_in_orthcompl(mat, dist):
    """Get a point in the orthogonal complement of the hyperplane containing
    all the points in the columns of mat, such that this point is mutually
    equidistant to all those points.

    If n == 1, then the method returns a 1-dimensional vector containing as
    entry mat[0,0]+dist.
    If n > 1, then the method first computes the normal vector of the
    hyperplane. This normalvector lies in the orthogonal complement, and can
    thus be computed as the null space of a matrix, that contains a basis of
    the hyperplane. Such a basis is constructed by considering all distance
    vectors for a a column vector in mat (except the first one) to the first
    column vector in mat. Furthermore, the correct point in the orthogonal
    complement is computed as follows. Let c be the centroid of the column
    vectors in mat. Then the return point p = c + gamma*w, where w is the
    normal vector of the hyperplane and gamma is chosen, such that the distance
    of p to all points in mat equals the parameter dist.

    Args:
        mat: An nxn matrix. The columns of the matrix will be considered as
            points laying in a hyperplane.
        dist: The distance, that the constructed point shall have to all other
            points.

    Returns:
        A point laying in the orthogonal complement of the hyerplane defined by
        mat, that has the distance dist to all column vectors of mat.
    """
    n, m = mat.shape
    assert(n == m)

    # Orthogonal vector to hyperplane.
    w = np.zeros(n)

    if n == 1:
        return np.ones((1,)) * (mat[0,0]+dist)

    # The column vectors of mat span a hyperplane (mat is nxn). Compute basis
    # of hyperplane.
    basis = np.zeros((n,n))
    reference = mat[:,0]

    for i in range(1,n):
        basis[:,i-1] = mat[:,i] - reference

    # The normal vector of a hyperplane is a basis for the orthogonal
    # complement, given by the null space of the basis of the hyperplane.
    w = lin_alg.compute_null_space(basis)
    w = w.squeeze()
    w /= np.linalg.norm(w)

    # Compute centroid.
    centroid = np.zeros(n)
    for i in range(n):
        centroid += mat[:,i]
    centroid /= n

    # Compute gamma such that:
    # || (centroid + gamma*w) - reference || == dist
    temp = centroid - reference
    # I.e., we have to solve:
    # 0 = gamma^2 * sum_i w_i^2 + gamma * 2 * sum_i w_i*temp_i
    #     + (-dist^2 + sum_i temp_i)

    # Using: a*gamma^2 + b*gamma + c = 0
    a = 0
    b = 0
    c = -(dist**2)
    for i in range(n):
        a += w[i]**2
        b += 2 * w[i] * temp[i]
        c += temp[i]**2

    gamma = (-b + np.sqrt(b**2 - 4*a*c)) / (2*a)

    return centroid + gamma*w

def _construct_simplex_iteratively(n,m,dist=1):
    """Construct an n-dimensional simplex iteratively, starting with a random
    point in the 1-dimensional space. All vertices of the simplex will be
    mutually equidistant.

    Note, this method is more computational expensive, then the algorithm
    implemented in the method '_construct_simplex_directly'. Additionally, due
    to the computational overhead, precision errors are more likely to occur
    and accumulate. Therefore, this the other method should usually always be
    preferred.

    The algorithm works as follows. At first a random point in the 1D space is
    picked. Then a simplex with two vertices (a line) in this space is
    contructed. This vertex is mapped into a 2D space, where a new vertex is
    added by constructing a simplex with 3 vertices (triangle). This process is
    repeated iteratively until the simplex has m vertices. If necessary, the
    simplex is finally mapped to the n-dimensional space.

    Args:
        n: The dimensionality of vertices in the constructed simplex.
        m: The number of vertices in the simplex (must be <= n+1).
        dist: The distance between two vertices of the constructed simplex.

    Returns:
        A nxm matrix, where each column contains a vertex of the simplex.
    """
    assert(m <= n+1)

    # Randomly pick first point
    k = 1 # current simplex dimensionality
    simplex = np.ones((1,1))
    while(k < min(n,m)):
        new_point = _get_mutequi_point_in_orthcompl(simplex, dist)
        k += 1
        new_simplex = np.zeros((k,k))
        new_simplex[:(k-1),:(k-1)] = simplex
        new_simplex[:(k-1),k-1] = new_point
        simplex = new_simplex

    if k < m:
        assert(k == n and k+1 == m)
        new_point = _get_mutequi_point_in_orthcompl(simplex, dist)
        new_simplex = np.zeros((n,n+1))
        new_simplex[:n,:n] = simplex
        new_simplex[:n,n] = new_point
        simplex = new_simplex
    else:
        new_simplex = np.zeros((n,m))
        new_simplex[:k,:k] = simplex
        simplex = new_simplex

    return simplex

def _construct_simplex_directly(n,m,dist=1):
    """Construct an n-dimensional simplex directly by starting from a canonical
    unit basis. All vertices of the simplex will be mutually equidistant.

    The algorithm works as follows. An identity matrix of size min(n,m) is
    generated. The column vectors are already valid vertices of a simplex, whit
    mutually equidistant vertices of distance sqrt(2). If m == n+1, then an
    additional point is added to this simplex, to construct a full
    n-dimensional simplex.

    Args:
        n: The dimensionality of vertices in the constructed simplex.
        m: The number of vertices in the simplex (must be <= n+1).
        dist: The distance between two vertices of the constructed simplex.

    Returns:
        A nxm matrix, where each column contains a vertex of the simplex.
    """
    assert(m <= n+1)

    simplex = np.identity(min(n,m))*dist/np.sqrt(2)

    if m > n:
        new_point = _get_mutequi_point_in_orthcompl(simplex, dist)
        new_simplex = np.zeros((n,n+1))
        new_simplex[:n,:n] = simplex
        new_simplex[:n,n] = new_point
        simplex = new_simplex
    elif m < n:
        new_simplex = np.zeros((n,m))
        new_simplex[:m,:m] = simplex
        simplex = new_simplex


    return simplex

def _move_simplex_in_unit_sphere(simplex):
    """Move and scale a simplex, such that all of its vertices lay on the unit
    sphere.

    Args:
        simplex: The simplex to be transformed.

    Returns:
        The transformed given simplex (in-place operation).
    """
    n, m = simplex.shape

    centroid = np.zeros(n)
    for i in range(m):
        centroid += simplex[:,i]
    centroid /= m

    for i in range(m):
        simplex[:,i] -= centroid
        simplex[:,i] /= np.linalg.norm(simplex[:,i])

    return simplex

def get_unit_simplex(n,m):
    """Get a unit simplex of dimensionality n with m vertices. We denote a
    unist simplex, a simplex whose vertices are mutually equidistant and they
    all lie on the surface of a unit sphere.

    Args:
        n: The dimensionality of the vertices in the constructed simplex.
        m: The number of vertices in the simplex (must be <= n+1).

    Returns:
        A nxm matrix whose column vectors span a unit simplex.
    """
    assert(m <= n+1)

    simplex = _construct_simplex_directly(n,m)
    simplex = _move_simplex_in_unit_sphere(simplex)

    return simplex

if __name__ == '__main__':
    pass
