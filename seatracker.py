# Copyright 2017-2018 The Susan Allen, Doug Latornell,
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
"""Post-processing particle tracker for NEMO ocean model run results.

Uses Runge-Kutta integration and 4d linear interpolation algorithms from
SciPy and includes handling of variable sea surface height produced by the
VVL option in NEMO.

Please see the TestSeaTrackerObject.ipynb notebook for an example of how to
use this module.
"""
from enum import IntEnum

import attr
import netCDF4
import numpy


class Dim4d(IntEnum):
    """Symbolic aliases for the dimension indices of NEMO 4d arrays.
    """
    t = 0
    z = 1
    y = 2
    x = 3


@attr.s
class _Grid:
    """Particle tracking grid.

    :param mesh_mask_file: Path and file name of the NEMO mesh mask file
                           to initialize the mesh from.
    :type mesh_mask_file: :py:class:`pathlib.Path` or str
    """

    mesh_mask_file = attr.ib()
    ##TODO: Add docstrings for attributes below
    t_mask = attr.ib(init=False, default=None)
    e1u = attr.ib(init=False, default=None)
    e2v = attr.ib(init=False, default=None)
    e3w0 = attr.ib(init=False, default=None)
    #: Particle tracking grid level depths 2d array.
    depth = attr.ib(init=False, default=None)

    def setup(self):
        """Set up the particle tracking grid by extracting the necessary grid
        arrays from the NEMO mesh mask, and calculate the particle grid level
        depths.
        """
        with netCDF4.Dataset(self.mesh_mask_file) as mesh_mask:
            self.t_mask = mesh_mask['tmask'][0]
            self.e1u = mesh_mask['e1u'][0]
            self.e2v = mesh_mask['e2v'][0]
            self.e3w0 = mesh_mask['e3w_0'][0]
            mbathy = mesh_mask['mbathy'][0]
            gdepw_0 = mesh_mask['gdepw_0'][0]
            self.depth = numpy.empty_like(mbathy)
            for i in range(self.t_mask.shape[1]):
                for j in range(self.t_mask.shape[2]):
                    self.depth[i, j] = gdepw_0[int(mbathy[i, j]), i, j]
