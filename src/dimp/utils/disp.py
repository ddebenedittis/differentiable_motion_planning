from cycler import cycler
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
        rotation in degree
        0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
        use this path for marker argument of plt.scatter
    scale : float
        multiply a argument of plt.scatter with this factor got get markers
        with the same size independent of their rotation.
        Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """

    arr = np.array([[0.1, 0.3], [0.1, -0.3], [1, 0], [0.1, 0.3]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))
    codes = [
        mpl.path.Path.MOVETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.LINETO,
        mpl.path.Path.CLOSEPOLY,
    ]
    arrow_head_marker = mpl.path.Path(arr, codes)

    return arrow_head_marker, scale


def init_matplotlib(palette: str = 'okabe-ito'):
    colors = get_colors(palette)

    lines = ['-', '--', '-', '--', '-', '--', '-', '--', '-', '--']
    lines = lines[:len(colors)]
    
    default_cycler = (
        cycler(color=colors) +
        cycler('linestyle', lines)
    )

    colors = list(default_cycler.by_key()['color'])

    textsize = 12
    labelsize = 12

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=textsize)
    plt.rc('ytick', labelsize=textsize)
    plt.rc('axes', labelsize=labelsize, prop_cycle=default_cycler)
    plt.rc('legend', fontsize=textsize)

    plt.rc("axes", grid=True, xmargin=0)
    plt.rc("grid", linestyle='dotted', linewidth=0.25)

    plt.rcParams['figure.constrained_layout.use'] = True
    
def get_colors(palette: str = 'okabe-ito'):
    """Get a list of colors based on the specified palette."""
    if palette == 'okabe-ito':
        return ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#000000']
    if palette == 'matlab':
        return ['#0072BD', '#D95319', '#EDB120', '#7E2F8E', '#77AC30', '#4DBEEE', '#A2142F']
    if palette == 'colorbrewer_1':
        return ['#E41A1C', '#377EB8', '#4DAF4A', '#984EA3', '#FF7F00', '#FFFF33', '#A65628', '#F781BF', '#999999']
    
    raise ValueError(f"Unknown palette: {palette}")
