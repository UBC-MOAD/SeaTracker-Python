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
class SeaTracker:
    """Post-processing particle tracker for NEMO ocean model run results.

    :param mesh_mask_file: Path and file name of the NEMO mesh mask file
                           to initialize the grid from.
    :type mesh_mask_file: :py:class:`pathlib.Path` or str

    :param u_field_file: Path and file name of the NEMO u velocity component
                         results field file to track particles in.
    :type u_field_file: :py:class:`pathlib.Path` or str

    :param v_field_file: Path and file name of the NEMO v velocity component
                         results field file to track particles in.
    :type v_field_file: :py:class:`pathlib.Path` or str

    :param w_field_file: Path and file name of the NEMO w velocity component
                         results field file to track particles in.
    :type w_field_file: :py:class:`pathlib.Path` or str
    """
    mesh_mask_file = attr.ib()
    u_field_file = attr.ib()
    v_field_file = attr.ib()
    w_field_file = attr.ib()

    #: Particle tracking grid; :py:class:`seatracker._Grid` instance.
    _grid = attr.ib(init=False, default=None)
    #: u component of velocity field in which to track particles;
    #: :py:class:`seatracker._ModelField` instance.
    _u_field = attr.ib(init=False, default=None)
    #: v component of velocity field in which to track particles;
    #: :py:class:`seatracker._ModelField` instance.
    _v_field = attr.ib(init=False, default=None)
    #: w component of velocity field in which to track particles;
    #: :py:class:`seatracker._ModelField` instance.
    _w_field = attr.ib(init=False, default=None)

    def setup(self):
        """Set up the particle tracker.

        1. Set up the particle tracking grid.
        2. Set up the model results velocity component fields to track
           particles in.
        """
        self._grid = _Grid(self.mesh_mask_file)
        self._grid.setup()

        self._u_field = _ModelField(self.u_field_path)
        self._v_field = _ModelField(self.v_field_path)
        fields = zip((self._u_field, self._v_field), ('depthu', 'depthv'),
                     ('vozocrtx', 'vomecrty'))
        for vel_field, depth_var, vel_var in fields:
            vel_field.load()
            vel_field.setup(depth_var)
            vel_field.values[:, 1:] = vel_field.dataset[vel_var][0:3]
            vel_field.values[:, 0] = (
                2 * vel_field.values[:, 1] - vel_field.values[:, 2]
            )
        self._u_field.coords[Dim4d.x] = self._u_field.coords[Dim4d.x] + 0.5
        self._v_field.coords[Dim4d.y] = self._v_field.coords[Dim4d.y] + 0.5

        self._w_field = _ModelField(self.w_field_path)
        self._w_field.load()
        self._w_field.setup('depthw')
        # Change to sign to positive velocity downward
        self._w_field.values = -self._w_field.dataset['vovecrtz'][0:3]
        self._w_field.coords[Dim4d.z] = self._w_field.coords[Dim4d.z] + 0.5


@attr.s
class _Grid:
    """Particle tracking grid.

    :param mesh_mask_file: Path and file name of the NEMO mesh mask file
                           to initialize the grid from.
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


@attr.s
class _ModelField:
    """NEMO model results field.

    :param model_field_file: Path and file name of the NEMO model results field
                             file to load.
    :type model_field_file: :py:class:`pathlib.Path` or str
    """

    model_field_file = attr.ib()
    #: Model results field :py:class:`netCDF4.Dataset` instance.
    dataset = attr.ib(init=False, default=None)
    #: Model results time step [s].
    delta_t = attr.ib(init=False, default=None)
    ##TODO: Get Susan to describe model results field coords array
    coords = attr.ib(init=False, default=None)
    #: Model results field values 4d array.
    values = attr.ib(init=False, default=None)

    def setup(self, depth_var):
        """Set up the model results field arrays.

        Note that the actual loading of the field values as well as finalization
        of the field coordinates are specific to different field variables,
        so must be done after this method is called.

        :param str depth_var: Name of the NEMO depth variable for the field.
        """
        self.dataset = netCDF4.Dataset(self.model_field_file)
        times = self.dataset['time_counter'][:3]
        self.delta_t = times[1] - times[0]
        x_coords = range(self.dataset.dimensions['x'].size)
        y_coords = range(self.dataset.dimensions['y'].size)
        depthsize = self.dataset[depth_var][:].shape[0]
        z_coords = numpy.linspace(0, depthsize, depthsize + 1) - 0.5
        longaxis = max(len(x_coords), len(y_coords), len(z_coords), len(times))
        self.coords = numpy.zeros((4, longaxis))
        self.coords[Dim4d.t, 0:len(times)] = times
        self.coords[Dim4d.t, len(times):] = max(times)
        self.coords[Dim4d.z, 0:len(z_coords)] = z_coords
        self.coords[Dim4d.z, len(z_coords):] = max(z_coords)
        self.coords[Dim4d.y, 0:len(y_coords)] = y_coords
        self.coords[Dim4d.y, len(y_coords):] = max(y_coords)
        self.coords[Dim4d.x, 0:len(x_coords)] = x_coords
        self.coords[Dim4d.x, len(x_coords):] = max(x_coords)
        self.values = numpy.zeros(
            (3, len(z_coords), len(y_coords), len(x_coords))
        )
