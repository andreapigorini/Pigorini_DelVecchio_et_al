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






def surface_fsav(coords, var, subjects_dir, vmin=None, vmax=None, cbar=True, scale=15,
                 hemis='both', surf='pial', cmap='viridis', surf_color='gray', hgl=None):
    """
    Visualize contact-level values on the fsaverage cortical surface.

    This function projects SEEG contact coordinates onto the FreeSurfer fsaverage surface
    and displays them as colored points in 3D using PyVista. Contact colors encode the
    selected variable, allowing visualization of response metrics such as responsiveness,
    offset, AUC, principal-gradient values, or anatomical/electrophysiological labels.
    Points can be displayed on the pial or inflated surface, either separately by hemisphere,
    jointly, or for a single hemisphere.

    Parameters
    ----------
    coords : pandas.DataFrame
        Contact-level dataframe containing fsaverage coordinates and the variable to plot.
        Required columns include x_norm_fsav, y_norm_fsav, z_norm_fsav, x_norm_surf,
        and the column specified by `var`.
    var : str
        Name of the column in `coords` used to color the contacts.
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR containing the fsaverage subject.
    vmin : float, optional
        Lower color scale limit. If None, the minimum value of `var` is used.
    vmax : float, optional
        Upper color scale limit. If None, the maximum value of `var` is used.
    cbar : bool, optional
        Whether to display the scalar bar.
    scale : float, optional
        Point size for contact visualization.
    hemis : {'both', 'joint', 'lh', 'rh'}, optional
        Hemisphere display mode.
    surf : {'pial', 'inflated'}, optional
        Surface type used for visualization.
    cmap : str or matplotlib colormap, optional
        Colormap used to encode `var`.
    surf_color : str, optional
        Base color of the cortical surface.
    hgl : list-like, optional
        Optional list of labels to highlight with larger markers.

    Returns
    -------
    plotter : pyvista.Plotter
        PyVista plotter object containing the rendered surface and contacts.
    """

    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import os.path as op
    import nibabel as nib
    import pyvista as pv
    from scipy.spatial.distance import cdist

    # Load fsaverage pial surfaces for both hemispheres.
    template = 'fsaverage'
    fname_pial = op.join(subjects_dir, template, 'surf', '%s.pial')
    pials = {h: nib.freesurfer.read_geometry(fname_pial % h) for h in ['lh', 'rh']}

    # Load inflated surfaces if requested.
    if surf == 'inflated':
        fname_inf = op.join(subjects_dir, template, 'surf', '%s.inflated')
        infl = {h: nib.freesurfer.read_geometry(fname_inf % h) for h in ['lh', 'rh']}

    # Convert FreeSurfer meshes into PyVista PolyData objects.
    load_hemis = ['lh', 'rh']
    pial_surfs = {}
    infl_surfs = {}

    for h in load_hemis:
        ndim_vect = np.repeat(3, len(pials[h][1])).reshape(-1, 1)
        pial_surfs[h] = pv.PolyData(pials[h][0], np.hstack([ndim_vect, pials[h][1]]))

        if surf == 'inflated':
            infl_surfs[h] = pv.PolyData(infl[h][0], np.hstack([ndim_vect, infl[h][1]]))

    # Set color scale limits.
    if vmin is None:
        vmin = coords[var].min()
    if vmax is None:
        vmax = coords[var].max()

    # Convert variable values into RGBA colors.
    cmap_obj = matplotlib.colormaps[cmap] if isinstance(cmap, str) else cmap
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    colors = cmap_obj(norm(coords[var]))
    coords[['r', 'g', 'b', 'a']] = colors

    # Split contacts by hemisphere.
    h_coords = {
        h: coords.loc[coords.x_norm_surf < 0] if h == 'lh'
        else coords.loc[coords.x_norm_surf > 0]
        for h in load_hemis
    }

    if surf == 'pial':

        if hemis == 'both':
            plotter = pv.Plotter(shape=(1, 2), notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot left and right hemispheres in separate panels.
            for ix, h in enumerate(load_hemis):
                plotter.subplot(0, ix)
                plotter.add_mesh(pial_surfs[h], opacity=0.05, color=surf_color)

                pts = plotter.add_points(
                    h_coords[h][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                    scalars=h_coords[h][['r', 'g', 'b', 'a']].values,
                    rgb=True,
                    render_points_as_spheres=True,
                    point_size=scale,
                    color='k',
                    cmap=cmap
                )

                # Optionally highlight selected anatomical/electrode labels.
                if hgl is not None:
                    hgl_coords = h_coords[h].query('label in @hgl')
                    if len(hgl_coords) != 0:
                        pts = plotter.add_points(
                            hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                            scalars=hgl_coords[['r', 'g', 'b', 'a']].values,
                            rgb=True,
                            render_points_as_spheres=True,
                            point_size=scale * 2,
                            color='k',
                            cmap=cmap
                        )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(mapper=pts.mapper)

        elif hemis == 'joint':
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot both pial hemispheres in the same 3D scene.
            for h in load_hemis:
                plotter.add_mesh(pial_surfs[h], opacity=0.05, color=surf_color)

            pts = plotter.add_points(
                coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                scalars=coords[['r', 'g', 'b', 'a']].values,
                rgb=True,
                render_points_as_spheres=True,
                point_size=scale,
                color='k',
                cmap=cmap
            )

            if hgl is not None:
                hgl_coords = coords.query('label in @hgl')
                pts = plotter.add_points(
                    hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                    scalars=hgl_coords[['r', 'g', 'b', 'a']].values,
                    rgb=True,
                    render_points_as_spheres=True,
                    point_size=scale * 2,
                    color='k',
                    cmap=cmap
                )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(mapper=pts.mapper)

        else:
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot contacts on a single pial hemisphere.
            plotter.add_mesh(pial_surfs[hemis], opacity=0.05, color=surf_color)

            pts = plotter.add_points(
                h_coords[hemis][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                scalars=h_coords[hemis][['r', 'g', 'b', 'a']].values,
                rgb=True,
                render_points_as_spheres=True,
                point_size=scale,
                color='k',
                cmap=cmap
            )

            if hgl is not None:
                hgl_coords = h_coords[hemis].query('label in @hgl')
                pts = plotter.add_points(
                    hgl_coords[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values,
                    scalars=hgl_coords[['r', 'g', 'b', 'a']].values,
                    rgb=True,
                    render_points_as_spheres=True,
                    point_size=scale * 2,
                    color='k',
                    cmap=cmap
                )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(mapper=pts.mapper)

    elif surf == 'inflated':

        # Map each pial contact coordinate to the closest pial vertex and retrieve
        # the corresponding vertex coordinate on the inflated surface.
        infl_coords = {h: [] for h in load_hemis}

        for h in load_hemis:
            foci_vtxs = np.argmin(
                cdist(
                    pials[h][0],
                    h_coords[h][['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values
                ),
                axis=0
            )
            infl_coords[h] = infl[h][0][foci_vtxs]

        if hemis == 'both':
            plotter = pv.Plotter(shape=(1, 2), notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot inflated hemispheres in separate panels.
            plotter.subplot(0, 0)
            plotter.add_mesh(infl_surfs['lh'], opacity=1, color=surf_color)
            pts = plotter.add_points(
                infl_coords['lh'],
                scalars=h_coords['lh'][['r', 'g', 'b', 'a']].values,
                rgb=True,
                render_points_as_spheres=True,
                point_size=scale,
                color='k',
                cmap=cmap
            )

            plotter.subplot(0, 1)
            plotter.add_mesh(infl_surfs['rh'], opacity=1, color=surf_color)
            pts = plotter.add_points(
                infl_coords['rh'],
                scalars=h_coords['rh'][['r', 'g', 'b', 'a']].values,
                rgb=True,
                render_points_as_spheres=True,
                point_size=scale,
                color='k',
                cmap=cmap
            )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(mapper=pts.mapper)

        elif hemis == 'joint':
            displacement = {'lh': -50, 'rh': 50}

            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot both inflated hemispheres in a displaced joint view.
            for h in load_hemis:
                plotter.add_mesh(
                    infl_surfs[h].translate([displacement[h], 0, 0]),
                    opacity=0.5,
                    color=surf_color
                )

                if len(infl_coords[h]) > 0:
                    infl_coords_disp = infl_coords[h].copy()
                    infl_coords_disp[:, 0] += displacement[h]

                    pts = plotter.add_points(
                        infl_coords_disp,
                        scalars=h_coords[h][['r', 'g', 'b', 'a']].values,
                        rgb=True,
                        render_points_as_spheres=True,
                        point_size=scale,
                        color='k',
                        cmap=cmap
                    )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(maper=pts.mapper)

        else:
            plotter = pv.Plotter(notebook=False)
            plotter.set_background('white')
            plotter.add_light(pv.Light(position=(1, 1, 1), color='white', intensity=0.01))

            # Plot contacts on a single inflated hemisphere.
            plotter.add_mesh(infl_surfs[hemis], opacity=1, color=surf_color)

            pts = plotter.add_points(
                infl_coords[hemis],
                scalars=h_coords[hemis][['r', 'g', 'b', 'a']].values,
                rgb=True,
                render_points_as_spheres=True,
                point_size=scale,
                color='k',
                cmap=cmap
            )

            pts.mapper.SetScalarRange(vmin, vmax)
            if cbar:
                plotter.add_scalar_bar(mapper=pts.mapper)

    plotter.show()

    return plotter






def flatmap_fsav(coords, var, subjects_dir, vmin=None, vmax=None, cbar=True, size=100,
                 alpha=1, cmap='plasma', marker='.',
                 atlas=None,
                 atlas_color='k',
                 atlas_lw=0.25,
                 atlas_alpha=0.4,
                 atlas_zorder=6,
                 atlas_smooth=True,
                 atlas_smooth_iter=2):
    """
    Visualize contact-level values on fsaverage cortical flatmaps.

    This function projects SEEG contacts from fsaverage 3D coordinates to the nearest
    cortical vertices and displays their values on pycortex flatmaps. The selected
    variable is shown as a scatter overlay on the cortical curvature map. Optional
    atlas contours can be added to provide anatomical or functional parcellation
    boundaries, including HCP-MMP1/Glasser or FreeSurfer annotations.

    Parameters
    ----------
    coords : pandas.DataFrame
        Contact-level dataframe containing fsaverage coordinates and the variable to plot.
        Required columns include x_norm_fsav, y_norm_fsav, z_norm_fsav, and `var`.
    var : str
        Name of the column in `coords` used to color the contacts.
    subjects_dir : str
        FreeSurfer SUBJECTS_DIR containing the fsaverage subject.
    vmin : float, optional
        Lower color scale limit. If None, the minimum value of `var` is used.
    vmax : float, optional
        Upper color scale limit. If None, the maximum value of `var` is used.
    cbar : bool, optional
        Whether to display a colorbar.
    size : float, optional
        Marker size for contact visualization.
    alpha : float, optional
        Marker transparency.
    cmap : str or matplotlib colormap, optional
        Colormap used to encode `var`.
    marker : str, optional
        Matplotlib marker style.
    atlas : str, optional
        Optional annotation/parcellation to display as flatmap contours.
        Examples include 'HCPMMP1', 'aparc', or 'aparc.a2009s'.
    atlas_color : str, optional
        Color of atlas contour lines.
    atlas_lw : float, optional
        Atlas contour linewidth.
    atlas_alpha : float, optional
        Atlas contour transparency.
    atlas_zorder : int, optional
        Z-order for atlas contours.
    atlas_smooth : bool, optional
        Whether to smooth atlas contour segments.
    atlas_smooth_iter : int, optional
        Number of smoothing iterations for atlas contours.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Matplotlib figure containing the fsaverage flatmap and contact overlay.
    """

    import mne
    import numpy as np
    import os.path as op
    import cortex
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from itcfpy.spatial import find_closest_vert

    def chaikin_smooth(polyline, n_iter=2):
        """
        Smooth a polyline using Chaikin's corner-cutting algorithm.
        """

        p = np.asarray(polyline, float)

        if p.shape[0] < 3:
            return p

        for _ in range(n_iter):
            Q = 0.75 * p[:-1] + 0.25 * p[1:]
            R = 0.25 * p[:-1] + 0.75 * p[1:]
            p = np.vstack([Q[0], np.column_stack((R[:-1], Q[1:])).reshape(-1, 2), R[-1]])

        return p

    def _label_boundary_segments(polys, flatpts2, label_vertices):
        """
        Extract flatmap line segments corresponding to the boundary of a cortical label.
        """

        in_label = np.zeros(len(flatpts2), dtype=bool)
        in_label[np.asarray(label_vertices, dtype=int)] = True

        # Identify triangle edges crossing the label boundary.
        tris = polys
        in_tri = in_label[tris]
        edges = np.stack([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]], axis=1).reshape(-1, 2)
        in_edge = np.stack([in_tri[:, [0, 1]], in_tri[:, [1, 2]], in_tri[:, [2, 0]]], axis=1).reshape(-1, 2)

        boundary = (in_edge.sum(axis=1) == 1)
        bedges = edges[boundary]
        bedges = np.sort(bedges, axis=1)
        bedges = np.unique(bedges, axis=0)

        return flatpts2[bedges]

    def _smooth_segments(segs, n_iter=2):
        """
        Smooth atlas boundary segments if requested.
        """

        out = []

        for s in segs:
            if atlas_smooth:
                out.append(chaikin_smooth(s, n_iter=n_iter))
            else:
                out.append(np.asarray(s, float))

        return out

    # Split contact dataframe by hemisphere.
    coords_l = coords.loc[coords.x_norm_surf < 0]
    coords_r = coords.loc[coords.x_norm_surf > 0]

    # Load fsaverage pial and white surfaces, then compute the mid-cortical surface.
    pial, white, fidu = {}, {}, {}

    for h in ['lh', 'rh']:
        pial[h] = mne.read_surface(op.join(subjects_dir, 'fsaverage', 'surf', f'{h}.pial'))
        white[h] = mne.read_surface(op.join(subjects_dir, 'fsaverage', 'surf', f'{h}.white'))
        fidu[h] = (white[h][0] + pial[h][0]) / 2, pial[h][1]

    # Assign contacts to hemispheres based on fsaverage x-coordinate.
    h_coords = {
        'lh': coords.loc[coords['x_norm_fsav'] < 0].copy(),
        'rh': coords.loc[coords['x_norm_fsav'] > 0].copy()
    }

    # Determine whether to plot one or both hemispheres.
    sides = 'lh' if (len(h_coords['lh']) > 0) and (len(h_coords['rh']) == 0) else 'rh' \
        if (len(h_coords['lh']) == 0) and (len(h_coords['rh']) > 0) else ['lh', 'rh']

    # Find the closest fsaverage surface vertex for each left-hemisphere contact.
    if 'lh' in sides:
        print('finding left hemisphere coords')
        h_coords['lh']['vert'] = h_coords['lh'].apply(
            lambda x: find_closest_vert(
                [x['x_norm_fsav'], x['y_norm_fsav'], x['z_norm_fsav']],
                surf=fidu['lh'][0]
            ),
            axis=1
        )
        selected_pts_l = h_coords['lh'].vert.values

    # Find the closest fsaverage surface vertex for each right-hemisphere contact.
    if 'rh' in sides:
        print('finding right hemisphere coords')
        h_coords['rh']['vert'] = h_coords['rh'].apply(
            lambda x: find_closest_vert(
                [x['x_norm_fsav'], x['y_norm_fsav'], x['z_norm_fsav']],
                surf=fidu['rh'][0]
            ),
            axis=1
        )
        selected_pts_r = h_coords['rh'].vert.values

    # Create an empty pycortex vertex object to display the fsaverage flatmap background.
    no_dat = np.empty(len(fidu['lh'][0]) + len(fidu['rh'][0]))
    no_dat[:] = np.nan
    vertex_no_dat = cortex.Vertex(no_dat, 'fsaverage')

    fig = cortex.quickflat.make_figure(
        vertex_no_dat,
        with_curvature=True,
        with_colorbar=False,
        with_rois=False
    )

    # Retrieve pycortex flatmap coordinates and triangulations.
    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf('fsaverage', "flat", nudge=True)
    lflat2 = lflatpts[:, :2] if lflatpts.shape[1] > 2 else lflatpts
    rflat2 = rflatpts[:, :2] if rflatpts.shape[1] > 2 else rflatpts

    # Set color scale limits.
    if vmin is None:
        vmin = coords[var].min()
    if vmax is None:
        vmax = coords[var].max()

    ax = plt.gca()

    # Optionally add atlas/parcellation contours.
    if atlas is not None:

        # Normalize common atlas aliases.
        if atlas.upper() in ("HCPMMP1", "HCP-MMP1", "GLASSER", "HCP"):
            mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
            parc = "HCPMMP1"
        else:
            parc = atlas

        def _add_hemi_contours(hemi, polys, flat2):
            """
            Add parcellation contours for a single hemisphere.
            """

            labels = mne.read_labels_from_annot(
                'fsaverage',
                parc=parc,
                hemi=hemi,
                subjects_dir=subjects_dir
            )

            all_polylines = []

            for lab in labels:
                if len(lab.vertices) == 0:
                    continue

                segs = _label_boundary_segments(polys, flat2, lab.vertices)
                all_polylines.extend(_smooth_segments(segs, n_iter=atlas_smooth_iter))

            if len(all_polylines):
                lc = LineCollection(
                    all_polylines,
                    colors=atlas_color,
                    linewidths=atlas_lw,
                    alpha=atlas_alpha,
                    zorder=atlas_zorder,
                    capstyle='round',
                    joinstyle='round'
                )
                ax.add_collection(lc)

        if 'lh' in sides:
            _add_hemi_contours('lh', lpolys, lflat2)

        if 'rh' in sides:
            _add_hemi_contours('rh', rpolys, rflat2)

    # Overlay left-hemisphere contacts on the flatmap.
    if 'lh' in sides:
        im = ax.scatter(
            lflat2[selected_pts_l, 0],
            lflat2[selected_pts_l, 1],
            s=size,
            c=coords_l[var],
            cmap=cmap,
            zorder=10,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            marker=marker
        )

    # Overlay right-hemisphere contacts on the flatmap.
    if 'rh' in sides:
        im = ax.scatter(
            rflat2[selected_pts_r, 0],
            rflat2[selected_pts_r, 1],
            s=size,
            c=coords_r[var],
            cmap=cmap,
            zorder=10,
            alpha=alpha,
            vmin=vmin,
            vmax=vmax,
            marker=marker
        )

    # Add horizontal colorbar.
    if cbar:
        cax = fig.add_axes([0.4, 0.9, 0.2, 0.025])
        fig.colorbar(im, cax=cax, orientation='horizontal')
        cax.set_title(var)

    fig.set_facecolor('w')

    return fig






def continuous_maps_one_cond(to_plot, correction, cmap, fs_dir, lims=[0, 0.25, 0.8],
                             sm=10, mode="weighted", transparent=True, distance=0.02,
                             th=0, cortex='white'):
    """
    Generate a continuous fsaverage surface map for one condition.

    This function projects contact-level values onto the fsaverage cortical surface
    using MNE's nearest-sensor interpolation. SEEG contacts are first represented as
    a subject-level montage in fsaverage MRI coordinates. The selected variable is then
    converted into an EvokedArray and interpolated to the cortical source space to obtain
    a continuous surface estimate. The resulting source estimate is visualized on a
    semi-inflated fsaverage surface.

    Parameters
    ----------
    to_plot : pandas.DataFrame
        Contact-level dataframe containing subject IDs, channel names, fsaverage
        coordinates, and the variable to project.
    correction : str
        Column name containing the contact-level values to be projected.
        This can be a binary responsiveness variable or a continuous metric.
    cmap : str or matplotlib colormap
        Colormap used for surface visualization.
    fs_dir : str
        FreeSurfer SUBJECTS_DIR containing the fsaverage subject and source space.
    lims : list, optional
        Three values defining the color scale limits for MNE plotting.
    sm : int, optional
        Number of smoothing steps applied during surface visualization.
    mode : str, optional
        Interpolation mode passed to mne.stc_near_sensors.
    transparent : bool, optional
        Whether to use transparency for zero or low values in the surface plot.
    distance : float, optional
        Maximum distance, in meters, used for nearest-sensor interpolation.
    th : float, optional
        Optional binarization threshold. If greater than zero, values below threshold
        are set to 0 and values above threshold are set to 1.
    cortex : str, optional
        Cortical surface color/style passed to MNE's stc.plot.

    Returns
    -------
    stc : mne.SourceEstimate
        Continuous surface source estimate obtained from contact-level values.
    brain : mne.viz.Brain
        MNE Brain visualization object.
    """

    # Create a unique subject-specific channel identifier.
    to_plot.loc[:, 'unique_ch_names'] = to_plot['subj'] + '_' + to_plot['ch_name']

    # Convert contact coordinates from millimeters to meters for MNE.
    ch_pos = {
        c.unique_ch_names: c[['x_norm_fsav', 'y_norm_fsav', 'z_norm_fsav']].values / 1000
        for ix, c in to_plot.iterrows()
    }

    # Build an MNE montage in fsaverage MRI coordinates.
    mont = mne.channels.make_dig_montage(ch_pos, coord_frame='mri')
    mont.add_estimated_fiducials('fsaverage')

    # Create an MNE info object matching the contact montage.
    info = mne.create_info(
        ch_names=mont.ch_names,
        ch_types=['seeg'] * len(mont.ch_names),
        sfreq=1000
    )
    info.set_montage(mont)

    # Store the contact-level variable as a single-sample EvokedArray.
    evo = mne.EvokedArray(to_plot[correction].values.reshape(-1, 1), info)

    # Load the fsaverage source space used for surface interpolation.
    src = mne.read_source_spaces(
        op.join(fs_dir, 'fsaverage', 'bem', "fsaverage-ico-5-src.fif")
    )

    # Interpolate contact-level values onto the cortical surface.
    stc = mne.stc_near_sensors(
        evo,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode=mode,
        distance=distance,
    )

    # Optionally binarize the continuous map.
    if th > 0:
        stc.data = np.where(stc.data < th, 0, 1)

    # Define color scaling for surface visualization.
    clim = dict(kind="value", lims=lims)

    # Plot the interpolated map on a semi-inflated fsaverage surface.
    brain = stc.plot(
        surface="pial_semi_inflated",
        cortex=cortex,
        hemi="split",
        colormap=cmap,
        colorbar=True,
        clim=clim,
        subjects_dir=fs_dir,
        size=(500, 500),
        smoothing_steps=sm,
        time_viewer=False,
        transparent=transparent,
        background='w',
        views=['lateral', 'medial']
    )

    return stc, brain






def pointplot_gamma_by_area(all_subj_coords, gamma_to_plot, variable, error='std'):
    """
    Plot area-wise gamma metrics grouped by cortical lobe.

    This function merges contact-level gamma metrics with anatomical labels and computes
    the mean value of a selected variable within each cortical area. Areas are sorted by
    decreasing mean value and displayed as a point plot, with colors indicating cortical
    lobes. Error bars can represent either standard deviation or standard error of the mean.

    Parameters
    ----------
    all_subj_coords : pandas.DataFrame
        Contact-level dataframe containing unique contact identifiers, cortical areas,
        and lobe labels.
    gamma_to_plot : pandas.DataFrame
        Contact-level dataframe containing unique contact identifiers and the gamma
        variable to summarize.
    variable : str
        Name of the gamma-derived variable to plot, such as AUC, offset, duration,
        or responsiveness.
    error : {'std', 'sem'}, optional
        Error-bar type. If 'std', standard deviation is plotted. If 'sem', standard
        error of the mean is plotted.

    Returns
    -------
    stats : pandas.DataFrame
        Area-wise summary table containing mean values, error estimates, and lobe labels.
    """

    # Merge gamma metrics with anatomical area and lobe labels.
    merged = pd.merge(
        gamma_to_plot[['unique_ch_names', variable]],
        all_subj_coords[['unique_ch_names', 'area', 'lobe']],
        on='unique_ch_names',
        how='inner'
    )

    # Stop if no contacts are shared between the two input dataframes.
    if merged.empty:
        raise ValueError("No overlapping unique_ch_names between gamma_to_plot and all_subj_coords.")

    # Standardize lobe names for plotting.
    merged['lobe'] = merged['lobe'].str.capitalize()

    # Define lobe-specific colors.
    palette = {
        'Occipital': (0, 0, 1),
        'Temporal': (0, 0.5, 0),
        'Parietal': (1, 0, 0),
        'Frontal': (0, 0, 0),
        'Insula': (0.5, 1, 0.5),
        'Cingulate': (1, 0.6, 1)
    }

    # Compute area-wise mean and error estimate.
    if error == 'sem':
        stats = (
            merged
            .groupby(['area', 'lobe'])[variable]
            .agg(mean='mean', err=lambda x: x.std(ddof=1) / np.sqrt(len(x)))
            .reset_index()
        )
        err_label = 'SEM'
    else:
        stats = (
            merged
            .groupby(['area', 'lobe'])[variable]
            .agg(mean='mean', err='std')
            .reset_index()
        )
        err_label = 'SD'

    # Truncate the lower bound of error bars at zero.
    stats['err_lower'] = np.maximum(0, stats['mean'] - stats['err'])
    stats['err_upper'] = stats['mean'] + stats['err']

    # Sort cortical areas by decreasing mean value.
    stats = stats.sort_values('mean', ascending=False)
    area_order = stats['area'].tolist()

    # Create area-wise point plot.
    plt.figure(figsize=(14, 6))

    sns.pointplot(
        data=stats,
        x='area',
        y='mean',
        hue='lobe',
        order=area_order,
        palette=palette,
        dodge=False,
        linestyle='none',
        markers='o',
        markersize=6,
        errorbar=None
    )

    # Add manually controlled error bars.
    for _, row in stats.iterrows():
        x = area_order.index(row['area'])
        y = row['mean']
        low = row['err_lower']
        high = row['err_upper']

        lower_err = abs(y - low) if y > low else 0
        upper_err = abs(high - y) if high > y else 0

        plt.errorbar(
            x=x,
            y=y,
            yerr=[[lower_err], [upper_err]],
            fmt='none',
            ecolor=palette.get(row['lobe'], 'gray'),
            capsize=4,
            alpha=0.9
        )

    # Mark zero reference level.
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Format axes and labels.
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Cortical area')
    plt.ylabel(f'{variable} (mean ± {err_label})')
    plt.title(f'{variable} by cortical area (mean ± {err_label}, sorted by mean)')
    plt.legend(title='Lobe', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    return stats