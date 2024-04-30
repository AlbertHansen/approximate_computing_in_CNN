import numpy as np
from scipy.spatial import distance
from skimage import color

def colorstr2rgb(c):
    rgbspec = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0, 0]]
    cspec = 'rgbwcmyk'
    k = cspec.find(c[0])
    if k != -1 and (k != 2 or len(c) == 1):
        return rgbspec[k]
    elif len(c) > 2:
        if c[:3].lower() == 'bla':
            return [0, 0, 0]
        elif c[:3].lower() == 'blu':
            return [0, 0, 1]
        else:
            raise ValueError("Unknown color string.")

def parsecolor(s):
    if isinstance(s, str):
        return colorstr2rgb(s)
    elif isinstance(s, list) and len(s) == 3:
        return s
    else:
        raise ValueError("Color specification cannot be parsed.")

def distinguishable_colors(n_colors, bg=[1, 1, 1], func=color.rgb2lab):
    if isinstance(bg, list) and isinstance(bg[0], list):
        bg = [parsecolor(c) for c in bg]
    else:
        bg = parsecolor(bg)

    n_grid = 30
    x = np.linspace(0, 1, n_grid)
    R, G, B = np.meshgrid(x, x, x)
    rgb = np.c_[R.ravel(), G.ravel(), B.ravel()]

    if n_colors > len(rgb) / 3:
        raise ValueError("You can't readily distinguish that many colors")

    lab = func(rgb)
    bglab = func(np.array([bg]))

    mindist2 = np.inf * np.ones(rgb.shape[0])
    for i in range(len(bglab) - 1):
        dX = lab - bglab[i]
        dist2 = np.sum(dX**2, axis=1)
        mindist2 = np.minimum(dist2, mindist2)

    colors = np.zeros((n_colors, 3))
    lastlab = bglab[-1]
    for i in range(n_colors):
        dX = lab - lastlab
        dist2 = np.sum(dX**2, axis=1)
        mindist2 = np.minimum(dist2, mindist2)
        index = np.argmax(mindist2)
        colors[i] = rgb[index]
        lastlab = lab[index]

    return colors