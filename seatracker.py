# Copyright 2017 The Susan Allen, Doug Latornell,
# and The University of British Columbia

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
"""
import itertools
import math
import random

import netCDF4
import numpy
from scipy.interpolate import LinearNDInterpolator


def initialize_mesh(mesh_mask_file):
    """
    ##TODO: finish doctring

    :param str mesh_mask_file: Path and file name of the NEMO mesh mask file
                               to initialize the mesh from.
    :return:
    :rtype: 6-tuple of :py:class:`numpy.ndarray`
    """
    ##TODO: Accept str, Path, or netCDF4.Dataset
    with netCDF4.Dataset(mesh_mask_file) as data:
        t_mask = data['tmask'][0]
        e1u = data['e1u'][0]
        e2v = data['e2v'][0]
        e3w0 = data['e3w_0'][0]
        mbathy = data['mbathy'][0]
        gdepw_0 = data['gdepw_0'][0]
        totaldepth = numpy.empty_like(mbathy)
        for i in range(t_mask.shape[1]):
            for j in range(t_mask.shape[2]):
                totaldepth[i, j] = gdepw_0[int(mbathy[i, j]), i, j]
        #TODO: Investigate runtime divide by zero and invalid value warnings
        fractiondepth = gdepw_0 / totaldepth
    ##TODO: Refactor to return SimpleNamespace
    return t_mask, e1u, e2v, e3w0, totaldepth, fractiondepth


def get_initial_data(u_field_path, v_field_path, w_field_path, tracer_fields_path, fractiondepth, totaldepth, e3w0):
    """
    ##TODO: finish doctring

    :param u_field_path: Path and file name of the NEMO u-velocity field to
                         load.

    :param v_field_path: Path and file name of the NEMO v-velocity field to
                         load.

    :param w_field_path: Path and file name of the NEMO w-velocity field to
                         load.

    :param tracer_fields_path: Path and file name of the NEMO tracer fields to
                               load.

    :param fractiondepth:
    :param totaldepth:
    :param e3w0:

    :return:
    :rtype: 15-tuple of :py:class:`numpy.ndarray`
    """
    ##TODO: Accept str, Path, or netCDF4.Dataset objects for field file paths
    udataset = netCDF4.Dataset(u_field_path)
    tcorrs = udataset['time_counter'][:3]
    deltat = tcorrs[1] - tcorrs[0]
    xcorrs = range(udataset.dimensions['x'].size)
    ycorrs = range(udataset.dimensions['y'].size)
    depthsize = udataset['depthu'][:].shape[0]
    # this makes it depth, not grid point and I need one above the surface
    zcorrs = numpy.linspace(0, depthsize, depthsize + 1) - 0.5

    longaxis = max(len(xcorrs), len(ycorrs), len(zcorrs), len(tcorrs))
    t_coords = numpy.zeros((4, longaxis))
    t_coords[0, 0:len(tcorrs)] = tcorrs
    t_coords[0, len(tcorrs):] = max(tcorrs)
    t_coords[1, 0:len(zcorrs)] = zcorrs
    t_coords[1, len(zcorrs):] = max(zcorrs)
    t_coords[2, 0:len(ycorrs)] = ycorrs
    t_coords[2, len(ycorrs):] = max(ycorrs)
    t_coords[3, 0:len(xcorrs)] = xcorrs
    t_coords[3, len(xcorrs):] = max(xcorrs)
    # other grids
    u_coords = numpy.copy(t_coords)
    u_coords[3] = t_coords[3] + 0.5
    v_coords = numpy.copy(t_coords)
    v_coords[2] = t_coords[2] + 0.5
    w_coords = numpy.copy(t_coords)
    w_coords[1] = t_coords[1] + 0.5

    u = numpy.zeros((3, len(zcorrs), len(ycorrs), len(xcorrs)))
    u[:, 1:] = udataset['vozocrtx'][0:3]
    u[:, 0] = 2 * u[:, 1] - u[:, 2]

    v = numpy.zeros_like(u)
    vdataset = netCDF4.Dataset(v_field_path)
    v[:, 1:] = vdataset['vomecrty'][0:3]
    v[:, 0] = 2 * v[:, 1] - v[:, 2]

    wdataset = netCDF4.Dataset(w_field_path)
    w = numpy.zeros_like(u)
    # Change to sign to positive velocity downward
    w = -wdataset['vovecrtz'][0:3]

    tdataset = netCDF4.Dataset(tracer_fields_path)
    ssh = tdataset['sossheig'][0:3]
    e3w = numpy.empty_like(w)
    for i in range(3):
        e3w[i] = e3w0 * (1 + ssh[i] / totaldepth)

    nextindex = 3
    ##TODO: Refactor to return SimpleNamespace?
    return u, v, w, tcorrs, t_coords, u_coords, v_coords, w_coords, deltat, nextindex, e3w, udataset, vdataset, wdataset, tdataset


def random_points(t_coords, deltat, t_mask):
    """Choose a random point in the domain at a random time before deltat
    and calculate a 3-D cloud of the 27 points nearest to (and including) the
    selected point.

    ##TODO: finish doctring

    :param t_coords:
    :param deltat:
    :param t_mask:

    :return:
    """
    t0 = t_coords[0, 0]
    yi = numpy.zeros((27, 3))

    good = 0
    while good == 0:
        tc = random.uniform(t0, t0+deltat)
        zc = random.uniform(0., 39.)
        yc = random.uniform(t_coords[2, 0], t_coords[2, -1])
        xc = random.uniform(t_coords[3, 0], t_coords[3, -1])

        count = 0
        for k in range(3):
            for j in range(3):
                for i in range(3):
                    yi[count, 0] = min(zc + k, 39.)
                    yi[count, 1] = min(yc + j, t_coords[2, -1])
                    yi[count, 2] = min(xc + i, t_coords[3, -1])
                    good += t_mask[math.floor(yi[count, 0]), math.floor(yi[count, 1]), math.floor(yi[count, 2])]
                    count += 1
    return tc, yi


def grid_points(t_coords, t_mask, zc, yc, xc):
    """Calculate the 3-D cloud of the 27 points nearest to (and including) the
    (zc, yc, xc) point.

    ##TODO: finish doctring

    :param t_coords:
    :param t_mask:
    :param zc:
    :param yc:
    :param xc:

    :return:
    """
    tc = t_coords[0, 0]
    yi = numpy.zeros((27, 3))

    good = 0
    count = 0
    for k in range(3):
        for j in range(3):
            for i in range(3):
                yi[count, 0] = min(zc + k, 39.)
                yi[count, 1] = min(yc + j, t_coords[2, -1])
                yi[count, 2] = min(xc + i, t_coords[3, -1])
                good += t_mask[math.floor(yi[count, 0]), math.floor(yi[count, 1]), math.floor(yi[count, 2])]
                ##TODO: Refactor to report points on land or bottom using
                ## something other than print()
                if t_mask[math.floor(yi[count, 0]), math.floor(yi[count, 1]), math.floor(yi[count, 2])] == 0:
                    print(zc+k, yc+j, xc+i)
                count += 1
    return tc, yi


def derivatives(t, poss, t_mask, e3w, e2v, e1u, w_coords, v_coords, u_coords, w, v, u):
    """

    ##TODO: finish doctring

    :param t:
    :param poss:
    :param t_mask:
    :param e3w:
    :param e2v:
    :param e1u:
    :param w_coords:
    :param v_coords:
    :param u_coords:
    :param w:
    :param v:
    :param u:

    :return:
    """
    rhs = numpy.zeros_like(poss)
    for ip in range(int(poss.shape[0]/3)):
        point = (t, poss[0 + ip * 3], poss[1 + ip * 3], poss[2 + ip * 3])
        rhs[0+ip*3:3+ip*3] = interpolator(
            t_mask, e3w, e2v, e1u, w_coords, v_coords, u_coords, w, v, u, point)
    return rhs


def interpolator(t_mask, e3w, e2v, e1u, w_coords, v_coords, u_coords, w, v, u, point):
    """

    Based on Stackoverflow https://stackoverflow.com/users/110026/jaime

    ##TODO: finish doctring

    :param t_mask:
    :param e3w:
    :param e2v:
    :param e1u:
    :param w_coords:
    :param v_coords:
    :param u_coords:
    :param w:
    :param v:
    :param u:
    :param point:

    :return:
    """
    rhs = numpy.zeros(3)
    if not t_mask[int(point[1]), int(point[2]), int(point[3])]:
        # point is on land
        return rhs
    vars = zip(
        (0, 1, 2), (e3w, e2v, e1u), (w_coords, v_coords, u_coords), (w, v, u))
    for vel_idx, scale, coord, vel in vars:
        indices, sub_coords = [], []
        good = True
        for j in range(len(point)):
            idx = numpy.digitize([point[j]], coord[j])[0]
            if idx == len(coord[j]):
                print(j, 'out of bounds', point[j], vel_idx, coord[j])
                good = False
            elif j == 1 and idx == 0:
                print(j, 'hit surface', point[j], vel_idx, coord[j])
                good = False
            else:
                indices.append([idx - 1, idx])
                sub_coords.append(coord[j][indices[-1]])
        if good:
            indices = numpy.array([j for j in itertools.product(*indices)])
            sub_coords = numpy.array([j for j in itertools.product(*sub_coords)])
            sub_data = vel[list(numpy.swapaxes(indices, 0, 1))]
            li = _construct_interpolator(sub_coords, sub_data)
            rhs[vel_idx] = _interpolate(li, point, indices, scale, vel_idx)
        else:
            rhs[vel_idx] = 0.
    return rhs


def _construct_interpolator(sub_coords, sub_data):
    li = LinearNDInterpolator(sub_coords, sub_data, rescale=True)
    return li


def _interpolate(li, point, indices, scale, vel_idx):
    if vel_idx == 0:
        rhs = li([point])[0] / scale[indices[0, 0], int(point[1]), int(point[2]), int(point[3])]
    else:
        rhs = li([point])[0] / scale[indices[0, 2], indices[0, 3]]
    return rhs


def update_arrays(totaldepth, e3w0, e3w, tcorrs, u_coords, v_coords, w_coords,
    u, v, w, deltat, nextindex, udataset, vdataset, wdataset, tdataset):
    """

    ##TODO: finish doctring

    :param totaldepth:
    :param e3w0:
    :param e3w:
    :param tcorrs:
    :param u_coords:
    :param v_coords:
    :param w_coords:
    :param u:
    :param v:
    :param w:
    :param deltat:
    :param nextindex:
    :param udataset:
    :param vdataset:
    :param wdataset:
    :param tdataset:

    :return:
    """
    tcorrs = tcorrs + deltat
    u_coords[0, 0:tcorrs.size] = tcorrs
    u_coords[0, tcorrs.size:] = tcorrs.max()
    v_coords[0] = u_coords[0]
    w_coords[0] = u_coords[0]
    u[0:2] = u[1:3]
    u[2, 1:] = udataset['vozocrtx'][nextindex]
    u[2, 0] = 2 * u[2, 1] - u[2, 2]
    v[0:2] = v[1:3]
    v[2, 1:] = vdataset['vomecrty'][nextindex]
    v[2, 0] = 2 * v[2, 1] - v[2, 2]
    w[0:2] = w[1:3]
    # Change to sign to positive velocity downward
    w[2] = -wdataset['vovecrtz'][nextindex]
    e3w[0:2] = e3w[1:3]
    e3w[2] = e3w0 * (1 + tdataset['sossheig'][nextindex] / totaldepth)
    nextindex += 1
    return tcorrs, u_coords, v_coords, w_coords, u, v, w, nextindex, e3w
