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
import math
import random

import netCDF4
import numpy


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
    #    xcorrs = udataset['gridX'][:]
    #    ycorrs = udataset['gridY'][:]
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
    w = - wdataset['vovecrtz'][
          0:3]  # change to velocity down (increasing depth)

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
