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
import netCDF4
import numpy


def initialize_mesh(mesh_mask_file):
    """

    :param str mesh_mask_file: Path and file name of the NEMO mesh mask file
                               to initialize the mesh from.
    :return:
    :rtype: 6-tuple of :py:class:`numpy.ndarray`
    """
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
        fractiondepth = gdepw_0 / totaldepth
    return t_mask, e1u, e2v, e3w0, totaldepth, fractiondepth
