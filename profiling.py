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
import numpy
from scipy.integrate import ode

import seatracker


mesh_mask_path = '../../MEOPAR/grid/mesh_mask201702.nc'
u_field_path = 'hindcast/SalishSea_1h_20160801_20160801_grid_U.nc'
v_field_path = 'hindcast/SalishSea_1h_20160801_20160801_grid_V.nc'
w_field_path = 'hindcast/SalishSea_1h_20160801_20160801_grid_W.nc'
tracer_fields_path = 'hindcast/SalishSea_1h_20160801_20160801_grid_T.nc'


def main():
    t_mask, e1u, e2v, e3w0, totaldepth, fractiondepth = \
        seatracker.initialize_mesh(
        mesh_mask_path)
    (u, v, w, tcorrs, t_coords, u_coords, v_coords, w_coords, deltat, nextindex,
     e3w, udataset, vdataset, wdataset, tdataset) = seatracker.get_initial_data(
        u_field_path, v_field_path, w_field_path, tracer_fields_path,
        fractiondepth, totaldepth, e3w0)
    tc, yi = seatracker.grid_points(t_coords, t_mask, 25-1, 446-1, 304-1)
    t0 = tcorrs[0]
    dt = deltat/10
    y0 = numpy.copy(yi)
    yp = numpy.ndarray.flatten(y0)
    integrator = ode(seatracker.derivatives).set_integrator('dopri5', atol=0.01)
    integrator.set_initial_value(yp, t0).set_f_params(
        t_mask, e3w, e2v, e1u, w_coords, v_coords, u_coords, w, v, u)
    # first segment
    t1 = t0 + 1.5*deltat
    while integrator.successful() and integrator.t < t1:
        integrator.integrate(integrator.t + dt)
    # more segments
    for count in range(1):
        # update arrays
        tcorrs, u_coords, v_coords, w_coords, u, v, w, nextindex, e3w = seatracker.update_arrays(
            totaldepth, fractiondepth, e3w0, e3w, tcorrs, u_coords, v_coords, w_coords, u, v, w,
            deltat, nextindex, udataset, vdataset, wdataset, tdataset)
        t1 += deltat
        while integrator.successful() and integrator.t < t1:
            integrator.integrate(integrator.t + dt)


if __name__ == '__main__':
    main()
