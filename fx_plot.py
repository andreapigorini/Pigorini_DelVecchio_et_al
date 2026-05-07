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






def continuous_maps_multi(to_plot, cmap, fs_dir, distance=0.015, surface='pial_semi_inflated', th=0.2):
    """
    Generate a multimodal continuous fsaverage surface map.

    This function projects acoustic, somatosensory, and visual contact-level
    responsiveness values onto the fsaverage cortical surface using MNE nearest-sensor
    interpolation. Each modality-specific map is thresholded and binarized, then combined
    into a single categorical multimodal map encoding the spatial overlap among sensory
    responses.

    The coding scheme is:
    0 = no response
    1 = acoustic only
    2 = somatosensory only
    3 = acoustic + somatosensory
    4 = visual only
    5 = acoustic + visual
    6 = somatosensory + visual
    7 = acoustic + somatosensory + visual

    Parameters
    ----------
    to_plot : pandas.DataFrame
        Contact-level dataframe containing subject IDs, channel names, fsaverage
        coordinates, and modality-specific responsiveness columns.
        Required columns are acoustic, somatosensory, visual, x_norm_fsav,
        y_norm_fsav, and z_norm_fsav.
    cmap : matplotlib colormap
        Discrete colormap used to display multimodal response categories.
    fs_dir : str, optional
        FreeSurfer SUBJECTS_DIR containing the fsaverage subject and source space.
    distance : float, optional
        Maximum distance, in meters, used for nearest-sensor interpolation.
    surface : str, optional
        Surface used for visualization. Typical options include 'pial_semi_inflated'
        and 'inflated'.
    th : float, optional
        Threshold applied to each interpolated modality map before binarization.

    Returns
    -------
    stc_all : mne.SourceEstimate
        Categorical source estimate encoding unimodal, bimodal, and trimodal responses.
    brain : mne.viz.Brain
        MNE Brain visualization object.
    """

    # Create a subject-specific unique contact identifier.
    to_plot.loc[:, 'unique_ch_names'] = to_plot['subj'] + '_' + to_plot['ch_name']

    # Convert fsaverage coordinates from millimeters to meters for MNE.
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

    # Load fsaverage source space for surface interpolation.
    src = mne.read_source_spaces(
        op.join(fs_dir, 'fsaverage', 'bem', "fsaverage-ico-5-src.fif")
    )

    # Project acoustic responsiveness onto the cortical surface and binarize it.
    evo_ac = mne.EvokedArray(to_plot['acoustic'].values.reshape(-1, 1), info)

    stc_ac = mne.stc_near_sensors(
        evo_ac,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode="weighted",
        distance=distance,
    )

    stc_ac.data = np.where(stc_ac.data < th, 0, 1)

    # Project somatosensory responsiveness onto the cortical surface and binarize it.
    evo_ss = mne.EvokedArray(to_plot['somatosensory'].values.reshape(-1, 1), info)

    stc_ss = mne.stc_near_sensors(
        evo_ss,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode="weighted",
        distance=distance,
    )

    stc_ss.data = np.where(stc_ss.data < th, 0, 1)

    # Project visual responsiveness onto the cortical surface and binarize it.
    evo_vi = mne.EvokedArray(to_plot['visual'].values.reshape(-1, 1), info)

    stc_vi = mne.stc_near_sensors(
        evo_vi,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode="weighted",
        distance=distance,
    )

    stc_vi.data = np.where(stc_vi.data < th, 0, 1)

    # Combine the three binary modality maps into a single categorical map.
    combined_bin = (
        stc_ac.data.astype(int) * 1 +
        stc_ss.data.astype(int) * 2 +
        stc_vi.data.astype(int) * 4
    )

    # Store the combined multimodal map as a SourceEstimate.
    stc_all = mne.SourceEstimate(
        combined_bin,
        vertices=stc_ac.vertices,
        tmin=stc_ac.tmin,
        tstep=stc_ac.tstep,
        subject=stc_ac.subject
    )

    # Plot on inflated surface and optionally add atlas borders.
    if surface == 'inflated':
        brain = stc_all.plot(
            surface="inflated",
            cortex='white',
            hemi="split",
            colormap=cmap,
            colorbar=False,
            clim=dict(kind="value", lims=[0, 3.5, 7]),
            subjects_dir=fs_dir,
            size=(500, 500),
            smoothing_steps='nearest',
            time_viewer=False,
            transparent=False,
            background='w',
            views=['lateral', 'medial']
        )

        labels = mne.read_labels_from_annot(
            subject='fsaverage',
            parc='aparc',
            hemi='both',
            subjects_dir=fs_dir
        )

        for label in labels:
            brain.add_label(
                label,
                borders=0.05,
                color='black',
                alpha=1
            )

    else:
        # Plot the categorical multimodal map on the selected fsaverage surface.
        brain = stc_all.plot(
            surface=surface,
            cortex='white',
            hemi="split",
            colormap=cmap,
            colorbar=True,
            clim=dict(kind="value", lims=[0, 3.5, 7]),
            subjects_dir=fs_dir,
            size=(500, 500),
            smoothing_steps='nearest',
            time_viewer=False,
            transparent=False,
            background='w',
            views=['lateral', 'medial']
        )

    return stc_all, brain






def continuous_maps_gamma_vs_lfp(to_plot, cmap, fs_dir):
    """
    Generate a continuous fsaverage surface map comparing gamma and LFP responsiveness.

    This function projects contact-level gamma and LFP responsiveness values onto the
    fsaverage cortical surface using MNE nearest-sensor interpolation. Each signal-specific
    map is thresholded and binarized, then combined into a categorical map encoding
    whether each surface vertex is supported by LFP-only, gamma-only, or overlapping
    gamma/LFP responsiveness.

    The coding scheme is:
    0 = no response
    1 = LFP only
    2 = gamma only
    3 = gamma + LFP

    Parameters
    ----------
    to_plot : pandas.DataFrame
        Contact-level dataframe containing unique contact identifiers, fsaverage
        coordinates, and gamma/LFP responsiveness columns.
        Required columns are unique_ch_names, x_norm_fsav, y_norm_fsav, z_norm_fsav,
        gamma, and lfp.
    cmap : matplotlib colormap
        Discrete colormap used to display gamma/LFP response categories.
    fs_dir : str
        FreeSurfer SUBJECTS_DIR containing the fsaverage subject and source space.

    Returns
    -------
    stc_all : mne.SourceEstimate
        Categorical source estimate encoding LFP-only, gamma-only, and overlapping
        gamma/LFP responsiveness.
    brain : mne.viz.Brain
        MNE Brain visualization object.
    """

    # Convert fsaverage contact coordinates from millimeters to meters for MNE.
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

    # Load the fsaverage source space used for surface interpolation.
    src = mne.read_source_spaces(
        op.join(fs_dir, 'fsaverage', 'bem', "fsaverage-ico-5-src.fif")
    )

    # Project gamma responsiveness onto the cortical surface and binarize it.
    evo_gamma = mne.EvokedArray(to_plot['gamma'].values.reshape(-1, 1), info)

    stc_gamma = mne.stc_near_sensors(
        evo_gamma,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode="weighted",
        distance=0.015,
    )

    stc_gamma.data = np.where(stc_gamma.data < 0.2, 0, 1)

    # Project LFP responsiveness onto the cortical surface and binarize it.
    evo_lfp = mne.EvokedArray(to_plot['lfp'].values.reshape(-1, 1), info)

    stc_lfp = mne.stc_near_sensors(
        evo_lfp,
        trans="fsaverage",
        subject="fsaverage",
        subjects_dir=fs_dir,
        src=src,
        surface="pial",
        mode="weighted",
        distance=0.015,
    )

    stc_lfp.data = np.where(stc_lfp.data < 0.2, 0, 1)

    # Combine binary LFP and gamma maps into a single categorical map.
    combined_bin = (
        stc_lfp.data.astype(int) * 1 +
        stc_gamma.data.astype(int) * 2
    )

    # Store the combined gamma/LFP map as a SourceEstimate.
    stc_all = mne.SourceEstimate(
        combined_bin,
        vertices=stc_gamma.vertices,
        tmin=stc_gamma.tmin,
        tstep=stc_gamma.tstep,
        subject=stc_gamma.subject
    )

    # Plot categorical gamma/LFP map on the semi-inflated fsaverage surface.
    brain = stc_all.plot(
        surface="pial_semi_inflated",
        cortex='white',
        hemi="split",
        colormap=cmap,
        colorbar=True,
        clim=dict(kind="value", lims=[0, 1.5, 3]),
        subjects_dir=fs_dir,
        size=(500, 500),
        smoothing_steps='nearest',
        time_viewer=False,
        transparent=True,
        background='w',
        views=['lateral', 'medial']
    )

    return stc_all, brain






def quickflat_with_atlas(
    vertex_or_vdat,
    subject=None,
    atlas='HCPMMP1',
    subjects_dir=None,
    with_curvature=True,
    with_colorbar=False,
    atlas_color='k',
    atlas_lw=0.25,
    atlas_alpha=0.35,
    atlas_smooth_iter=2,
    figsize=None,
    show_labels=False,
    label_filter=None,
    label_color='k',
    label_size=6,
    label_alpha=0.8,
    label_zorder=20,
    label_outline=True,
):
    """
    Render a pycortex quickflat visualization with atlas boundaries and optional labels.

    This function extends pycortex.quickflat by overlaying cortical parcellation
    contours (e.g., HCPMMP1 or FreeSurfer atlases) on the flattened fsaverage surface.
    It also supports optional annotation of region names directly on the flatmap.

    Parameters
    ----------
    vertex_or_vdat : cortex.Vertex or array-like
        Input data. Can be a pycortex Vertex object (e.g., Vertex, VertexRGB)
        or a 1D array of values defined on the cortical surface.
    subject : str, optional
        Subject identifier (required if passing a raw array).
    atlas : str, optional
        Atlas/parcellation name. Supports 'HCPMMP1' (Glasser) or any FreeSurfer
        annotation (e.g., 'aparc', 'aparc.a2009s').
    subjects_dir : str, optional
        FreeSurfer SUBJECTS_DIR, required for atlas loading.
    with_curvature : bool, optional
        Whether to show cortical curvature in the background.
    with_colorbar : bool, optional
        Whether to display the colorbar.
    atlas_color : str or tuple, optional
        Color of atlas boundaries.
    atlas_lw : float, optional
        Line width of atlas contours.
    atlas_alpha : float, optional
        Transparency of atlas contours.
    atlas_smooth_iter : int, optional
        Number of Chaikin smoothing iterations for contour lines.
    figsize : tuple, optional
        Figure size.
    show_labels : bool, optional
        Whether to display atlas region names.
    label_filter : str or list, optional
        Filter for labels (regex string or list of substrings).
    label_color : str, optional
        Text color for labels.
    label_size : int, optional
        Font size for labels.
    label_alpha : float, optional
        Transparency for labels.
    label_zorder : int, optional
        Z-order for labels.
    label_outline : bool, optional
        Whether to draw a white outline around text labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The rendered quickflat figure with atlas overlays.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    import matplotlib.patheffects as pe
    import mne
    import cortex
    import re

    # -------------------- helpers --------------------
    def _label_boundary_segments(polys, flatpts2, label_vertices):
        """Extract boundary edges of a label on the flat surface."""
        in_label = np.zeros(len(flatpts2), dtype=bool)
        in_label[np.asarray(label_vertices, dtype=int)] = True

        tris = polys
        in_tri = in_label[tris]

        edges = np.stack(
            [tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]],
            axis=1
        ).reshape(-1, 2)

        in_edge = np.stack(
            [in_tri[:, [0, 1]], in_tri[:, [1, 2]], in_tri[:, [2, 0]]],
            axis=1
        ).reshape(-1, 2)

        boundary = (in_edge.sum(axis=1) == 1)
        bedges = edges[boundary]
        bedges = np.sort(bedges, axis=1)
        bedges = np.unique(bedges, axis=0)

        return flatpts2[bedges]

    def _chaikin(polyline, n_iter=2):
        """Smooth polyline using Chaikin subdivision."""
        p = np.asarray(polyline, float)
        if p.shape[0] < 3:
            return p

        for _ in range(n_iter):
            Q = 0.75 * p[:-1] + 0.25 * p[1:]
            R = 0.25 * p[:-1] + 0.75 * p[1:]
            p = np.vstack([
                Q[0],
                np.column_stack((R[:-1], Q[1:])).reshape(-1, 2),
                R[-1]
            ])
        return p

    def _keep_label(name):
        """Filter labels based on substring or regex."""
        if label_filter is None:
            return True
        if isinstance(label_filter, str):
            return re.search(label_filter, name) is not None
        return any(s in name for s in label_filter)

    # -------------------- input handling --------------------
    tname = type(vertex_or_vdat).__name__
    tmod = type(vertex_or_vdat).__module__

    if hasattr(vertex_or_vdat, "subject") and ("Vertex" in tname) and tmod.startswith("cortex"):
        vtx = vertex_or_vdat
        if subject is None:
            subject = vertex_or_vdat.subject
    else:
        arr = np.asarray(vertex_or_vdat).squeeze()
        if arr.ndim != 1:
            raise ValueError(f"Input must be 1D after squeeze, got shape={arr.shape}")
        if subject is None:
            raise ValueError("You must specify `subject` when passing a raw array.")
        vtx = cortex.Vertex(arr, subject)

    # -------------------- base quickflat --------------------
    fig = cortex.quickflat.make_figure(
        vtx,
        with_curvature=with_curvature,
        with_colorbar=with_colorbar,
        with_rois=False
    )

    if figsize is not None:
        fig.set_size_inches(*figsize)

    ax = plt.gca()

    # -------------------- flat surfaces --------------------
    (lflatpts, lpolys), (rflatpts, rpolys) = cortex.db.get_surf(subject, "flat", nudge=True)

    lflat2 = lflatpts[:, :2] if lflatpts.shape[1] > 2 else lflatpts
    rflat2 = rflatpts[:, :2] if rflatpts.shape[1] > 2 else rflatpts

    # -------------------- atlas loading --------------------
    if atlas.upper() in ("HCPMMP1", "HCP-MMP1", "GLASSER", "HCP"):
        if subjects_dir is None:
            raise ValueError("HCPMMP1 requires subjects_dir.")
        mne.datasets.fetch_hcp_mmp_parcellation(subjects_dir=subjects_dir, verbose=False)
        parc = "HCPMMP1"
    else:
        parc = atlas

    labels_lh = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='lh', subjects_dir=subjects_dir)
    labels_rh = mne.read_labels_from_annot('fsaverage', parc=parc, hemi='rh', subjects_dir=subjects_dir)

    # -------------------- plotting --------------------
    def add_hemi(labels, polys, flat2, hemi_prefix):
        polylines = []

        for lab in labels:
            if len(lab.vertices) == 0:
                continue

            name = lab.name
            if not _keep_label(name):
                continue

            # Contours
            segs = _label_boundary_segments(polys, flat2, lab.vertices)
            for s in segs:
                polylines.append(_chaikin(s, atlas_smooth_iter) if atlas_smooth_iter else s)

            # Label text
            if show_labels:
                pts = flat2[np.asarray(lab.vertices, int)]
                if pts.size == 0:
                    continue

                x, y = np.nanmedian(pts[:, 0]), np.nanmedian(pts[:, 1])
                txt = ax.text(
                    x, y,
                    name.replace(f"-{hemi_prefix}", ""),
                    color=label_color,
                    fontsize=label_size,
                    alpha=label_alpha,
                    ha='center',
                    va='center',
                    zorder=label_zorder
                )

                if label_outline:
                    txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

        if polylines:
            lc = LineCollection(
                polylines,
                colors=atlas_color,
                linewidths=atlas_lw,
                alpha=atlas_alpha,
                zorder=10,
                capstyle='round',
                joinstyle='round'
            )
            ax.add_collection(lc)

    add_hemi(labels_lh, lpolys, lflat2, "lh")
    add_hemi(labels_rh, rpolys, rflat2, "rh")

    return fig