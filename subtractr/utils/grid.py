import numpy as np
from typing import NamedTuple, Any

# Convenience class for representing a grid of stim points

class StimGrid(NamedTuple):
    xs: Any
    ys: Any
    zs: Any
    xgrid: Any
    ygrid: Any
    zgrid: Any
    dims: Any
    points: Any

    def __hash__(self):
        return hash((self.xlims, self.ylims, self.zlims))
    
    def __eq__(self, other):
            return np.all(self.xs == other.xs) \
                and np.all(self.ys == other.ys) \
                and np.all(self.zs == other.zs)

def make_default_grid():
    return make_manual_grid(
        xlims=(194, 319),
        ylims=(194, 319),
        zlims=(-75, 25),
        radial_spacing=5,
        axial_spacing=25
    )
        

def make_manual_grid(xlims, ylims, zlims, radial_spacing, axial_spacing):
    assert len(xlims) == 2
    assert len(ylims) == 2
    assert len(zlims) == 2

    # make sure everything is a devicearray
    xlims, ylims, zlims = [np.array(x) for x in [xlims, ylims, zlims]]

    # calculate the number of grid points in each dimension
    xrange, yrange, zrange = [np.ptp(lim) for lim in [xlims, ylims, zlims]]
    xsteps, ysteps, zsteps = [int(np.round(r / spacing)) for (r, spacing) in \
        zip([xrange, yrange, zrange], [radial_spacing, radial_spacing, axial_spacing])]

    xs, ys, zs = [np.linspace(lim[0], lim[1], steps) for (lim, steps) in \
        zip([xlims, ylims, zlims], [xsteps, ysteps, zsteps])]

    xgrid, ygrid, zgrid = np.meshgrid(xs, ys, zs)
    dims = [len(arr) for arr in [xs, ys, zs]]
    points = np.c_[xgrid.flatten(), ygrid.flatten(), zgrid.flatten()]

    return StimGrid(
        xs, ys, zs, 
        xgrid, ygrid, zgrid,
        dims, points
    )

def make_grid_from_stim_locs(stim_locs):
    xs, ys, zs = [sorted(np.unique(x)) \
        for x in (stim_locs[:,0], stim_locs[:,1], stim_locs[:,2])]
    xlims = (xs[0], xs[-1])
    ylims = (ys[0], ys[-1])
    zlims = (zs[0], zs[-1])

    xgrid, ygrid, zgrid = np.meshgrid(xs, ys, zs)
    dims = [len(arr) for arr in [xs, ys, zs]]
    points = np.c_[xgrid.flatten(), ygrid.flatten(), zgrid.flatten()]

    return StimGrid(
        xs, ys, zs, 
        xgrid, ygrid, zgrid,
        dims, points
    )
