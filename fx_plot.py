from pymatreader import read_mat
import os
from itcfpy.process import remove_stim_artifact
from itcfpy.spatial import make_bip_lists, mni2fsav_coords
from scipy.stats import ttest_ind
from scipy import stats
from statsmodels.stats.multitest import multipletests
import pickle
import pandas as pd
import os.path as op
import numpy as np
import mne
import cortex
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from cortex.polyutils import Surface
from itcfpy.spatial import find_closest_vert
from scipy.sparse import csr_matrix
from mne.stats import permutation_cluster_1samp_test
from functools import reduce
import seaborn as sns



def surface_fsav(coords, var, subjects_dir, vmin=None, vmax=None, cbar=True, scale=15, hemis='both', surf='pial',
            cmap='viridis', surf_color='gray', hgl=None):
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import os.path as op
    import nibabel as nib
    import pyvista as pv
    from scipy.spatial.distance import cdist

    template = 'fsaverage'
    fname_pial = op.join(subjects_dir, template, 'surf', '%s.pial')
    pials = {h: nib.freesurfer.read_geometry(fname_pial % h) for h in ['lh', 'rh']}

    if surf == 'inflated':
        fname_inf = op.join(subjects_dir, template, 'surf', '%s.inflated')
        infl = {h: nib.freesurfer.read_geometry(fname_inf % h) for h in ['lh', 'rh']}

    load_hemis = ['lh', 'rh']
    pial_surfs = {}
    infl_surfs = {}
    for h in load_hemis:
        ndim_vect = np.repeat(3, len(pials[h][1])).reshape(-1, 1)
        pial_surf = pv.PolyData(pials[h][0], np.hstack([ndim_vect, pials[h][1]]))
        pial_surfs[h] = pial_surf

        if surf == 'inflated':
            infl_surf = pv.PolyData(infl[h][0], np.hstack([ndim_vect, infl[h][1]]))
            infl_surfs[h] = infl_surf

    if vmin == None:
        vmin = coords[var].min()
    if vmax == None:
        vmax = coords[var].max()

    if isinstance(cmap, str):
        cmap_obj = matplotlib.colormaps[cmap]
    else:
        cmap_obj = cmap  # assume it's already a colormap object


    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(coords[var]))
    coords[['r', 'g', 'b', 'a']] = colors

    h_coords = {h: coords.loc[coords.x_norm_surf < 0] if h == 'lh' else coords.loc[coords.x_norm_surf > 0] for h in load_hemis}

    if surf == 'pial':
        if hemis == 'both':
            plotter = pv.Plotter(shape=(1, 2), notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            for ix, h in enumerate(load_hemis):
                plotter.subplot(0, ix)
                plotter.add_mesh(pial_surfs[h], opacity=0.05, color=surf_color)
                pts = plotter.add_points(h_coords[h][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                                         scalars=h_coords[h][['r', 'g', 'b', 'a']].values, rgb=True,
                                         render_points_as_spheres=True, point_size=scale, color='k', cmap=cmap)
                if hgl is not None:
                    hgl_coords = h_coords[h].query('label in @hgl')
                    if len(hgl_coords) != 0:
                        pts = plotter.add_points(hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                                    scalars=hgl_coords[['r', 'g', 'b', 'a']].values, rgb=True,
                                    render_points_as_spheres=True, point_size=scale*2, color='k', cmap=cmap)

            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()
        elif hemis == 'joint':
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            for ix, h in enumerate(load_hemis):
                plotter.add_mesh(pial_surfs[h], opacity=0.05, color=surf_color)
                pts = plotter.add_points(coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                                         scalars=coords[['r', 'g', 'b', 'a']].values, rgb=True,
                                         render_points_as_spheres=True, point_size=scale, color='k', cmap=cmap)
                if hgl is not None:
                    hgl_coords = coords.query('label in @hgl')
                    pts = plotter.add_points(hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                                scalars=hgl_coords[['r', 'g', 'b', 'a']].values, rgb=True,
                                render_points_as_spheres=True, point_size=scale*2, color='k', cmap=cmap)

            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()
        else:
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            plotter.add_mesh(pial_surfs[hemis], opacity=0.05, color=surf_color)
            pts = plotter.add_points(h_coords[hemis][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                                     scalars=h_coords[hemis][['r', 'g', 'b', 'a']].values, rgb=True,
                                     render_points_as_spheres=True, point_size=scale, color='k', cmap=cmap)
            if hgl is not None:
                hgl_coords = h_coords[hemis].query('label in @hgl')
                pts = plotter.add_points(hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                            scalars=hgl_coords[['r', 'g', 'b', 'a']].values, rgb=True,
                            render_points_as_spheres=True, point_size=scale*2, color='k', cmap=cmap)

            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()

    elif surf == 'inflated':
        infl_coords = {h: [] for h in load_hemis}

        hemi_get = load_hemis if hemis == 'both' else [hemis]
        for h in load_hemis:
            foci_vtxs = np.argmin(cdist(pials[h][0], h_coords[h][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values), axis=0)
            infl_coords[h] = infl[h][0][foci_vtxs]

        if hemis == 'both':
            plotter = pv.Plotter(shape=(1, 2), notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            plotter.subplot(0, 0)
            plotter.add_mesh(infl_surfs['lh'], opacity=1, color=surf_color)
            pts = plotter.add_points(infl_coords['lh'], scalars=h_coords['lh'][['r', 'g', 'b', 'a']].values, rgb=True,
                                     render_points_as_spheres=True, point_size=scale, color='k',  cmap=cmap)

            plotter.subplot(0, 1)
            plotter.add_mesh(infl_surfs['rh'], opacity=1, color=surf_color)
            pts = plotter.add_points(infl_coords['rh'], scalars=h_coords['rh'][['r', 'g', 'b', 'a']].values, rgb=True,
                               render_points_as_spheres=True, point_size=scale, color='k',  cmap=cmap)

            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()

        elif hemis == 'joint':
            displacement = {'lh': -50, 'rh': 50}

            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            for ix, h in enumerate(load_hemis):
                plotter.add_mesh(infl_surfs[h].translate([displacement[h], 0, 0]), opacity=0.5, color=surf_color)
                if len(infl_coords[h]) > 0:
                    infl_coords_disp = infl_coords[h].copy()
                    infl_coords_disp[:, 0] += displacement[h]
                    pts = plotter.add_points(infl_coords_disp,
                                             scalars=h_coords[h][['r', 'g', 'b', 'a']].values, rgb=True,
                                             render_points_as_spheres=True, point_size=scale, color='k', cmap=cmap)
            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()
        else:
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            light = pv.Light(position=(1, 1, 1), color='white', intensity=0.01)
            plotter.add_light(light)

            plotter.add_mesh(infl_surfs[hemis], opacity=1, color=surf_color)
            pts = plotter.add_points(infl_coords[hemis], scalars=h_coords[hemis][['r', 'g', 'b', 'a']].values, rgb=True,
                                     render_points_as_spheres=True, point_size=scale, color='k', cmap=cmap)
            pts.mapper.SetScalarRange(vmin, vmax)
            plotter.add_scalar_bar(mapper=pts.mapper)
            #plotter.show()
    plotter.show()
    return plotter
