import time
import numpy as np



def timetest(input_func):
    def timed(*args, **kwargs):
        start_time = time.time()
        result = input_func(*args, **kwargs)
        end_time = time.time()
        # print("Method Name - {0}, Args - {1}, Kwargs - {2}, Execution Time - {3}".format(input_func.__name__, args, kwargs,
        #                                                                                  end_time - start_time))
        print("Method Name - {0}, Execution Time - {1}".format(input_func.__name__, end_time - start_time))
        return result

    return timed

######################
## sunpy map related##
######################

def map2wcsgrids(snpmap, cell=True):
    '''

    :param snpmap:
    :param cell: if True, return the coordinates of the pixel centers. if False, return the coordinates of the pixel boundaries
    :return:
    '''
    import astropy.units as u
    if cell:
        ny, nx = snpmap.data.shape
        offset = 0.0
    else:
        ny, nx = snpmap.data.shape
        nx += 1
        ny += 1
        offset = -0.5
    XX, YY = np.meshgrid(np.arange(nx) + offset, np.arange(ny) + offset)
    mesh = snpmap.pixel_to_world(XX * u.pix, YY * u.pix)
    mapx, mapy = mesh.Tx.value, mesh.Ty.value
    return mapx, mapy

#######################
##SDO/AIA map related##
#######################

def sdo_aia_scale_dict(wavelength=None, imagetype='image'):
    '''
    rescale the aia image
    :param image: normalised aia image data
    :param wavelength:
    :return: byte scaled image data
    '''
    if type(wavelength) is not str:
        wavelength = '{:0.0f}'.format(wavelength)
    if wavelength == '94':
        if imagetype == 'image':
            return {'low': 0.1, 'high': 3000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -30, 'high': 30, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -30, 'high': 30, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '131':
        if imagetype == 'image':
            return {'low': 0.5, 'high': 10000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -100, 'high': 100, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -100, 'high': 100, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '171':
        if imagetype == 'image':
            return {'low': 20, 'high': 5000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -400, 'high': 400, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -400, 'high': 400, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '193':
        if imagetype == 'image':
            return {'low': 30, 'high': 5000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -1500, 'high': 1500, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -1500, 'high': 1500, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '211':
        if imagetype == 'image':
            return {'low': 10, 'high': 8000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -300, 'high': 300, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -300, 'high': 300, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '304':
        if imagetype == 'image':
            return {'low': 1, 'high': 10000, 'log': True}  # return {'low': 1, 'high': 500, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -300, 'high': 300, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -300, 'high': 300, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '335':
        if imagetype == 'image':
            return {'low': 0.1, 'high': 800, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -15, 'high': 15, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -15, 'high': 15, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '1600':
        if imagetype == 'image':
            return {'low': 20, 'high': 2500, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -800, 'high': 800, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -800, 'high': 800, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    elif wavelength == '1700':
        if imagetype == 'image':
            return {'low': 300, 'high': 10000, 'log': True}
        elif imagetype == 'RDimage':
            return {'low': -1500, 'high': 1500, 'log': False}
        elif imagetype == 'BDimage':
            return {'low': -1500, 'high': 1500, 'log': False}
        elif imagetype == 'RDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
        elif imagetype == 'BDRimage':
            return {'low': -1.5, 'high': 1.5, 'log': False}
    else:
        return None

def normalize_aiamap(aiamap):
    '''
    do expisure normalization of an aia map
    :param aia map made from sunpy.map:
    :return: normalised aia map
    '''
    import sunpy.map as smap
    try:
        if aiamap.observatory == 'SDO' and aiamap.instrument[0:3] == 'AIA':
            data = aiamap.data
            data[~np.isnan(data)] = data[~np.isnan(data)] / aiamap.exposure_time.value
            data[data < 0] = 0
            aiamap.meta['exptime'] = 1.0
            aiamap = smap.Map(data, aiamap.meta)
            return aiamap
        else:
            raise ValueError('input sunpy map is not from aia.')
    except:
        raise ValueError('check your input map. There are some errors in it.')

###############################
## time-distance plot related##
###############################

def findDist(x, y):
    dx = np.diff(x)
    dy = np.diff(y)
    dist = np.hypot(dx, dy)
    return np.insert(dist, 0, 0.0)


def paramspline(x, y, length, s=0):
    from scipy.interpolate import splev, splprep
    tck, u = splprep([x, y], s=s)
    unew = np.linspace(0, u[-1], length)
    out = splev(unew, tck)
    xs, ys = out[0], out[1]
    grads = get_curve_grad(xs, ys)
    return {'xs': xs, 'ys': ys, 'grads': grads['grad'], 'posangs': grads['posang']}


def polyfit(x, y, length, deg):
    xs = np.linspace(np.nanmin(x), np.nanmax(x), length)
    z = np.polyfit(x=x, y=y, deg=deg)
    p = np.poly1d(z)
    ys = p(xs)
    grads = get_curve_grad(xs, ys)
    return {'xs': xs, 'ys': ys, 'grads': grads['grad'], 'posangs': grads['posang']}


def spline(x, y, length, s=0):
    from scipy.interpolate import splev, splrep
    xs = np.linspace(x.min(), x.max(), length)
    tck = splrep(x, y, s=s)
    ys = splev(xs, tck)
    grads = get_curve_grad(xs, ys)
    return {'xs': xs, 'ys': ys, 'grads': grads['grad'], 'posangs': grads['posang']}


def get_curve_grad(x, y):
    '''
    get the grad of at data point
    :param x:
    :param y:
    :return: grad,posang
    '''
    deltay = np.roll(y, -1) - np.roll(y, 1)
    deltay[0] = y[1] - y[0]
    deltay[-1] = y[-1] - y[-2]
    deltax = np.roll(x, -1) - np.roll(x, 1)
    deltax[0] = x[1] - x[0]
    deltax[-1] = x[-1] - x[-2]
    grad = deltay / deltax
    posang = np.arctan2(deltay, deltax)
    return {'grad': grad, 'posang': posang}


def improfile(z, xi, yi, interp='cubic'):
    '''
    Pixel-value cross-section along line segment in an image
    :param z: an image array
    :param xi and yi: equal-length vectors specifying the pixel coordinates of the endpoints of the line segment
    :param interp: interpolation type to sampling, 'nearest' or 'cubic'
    :return: the intensity values of pixels along the line
    '''
    import scipy.ndimage
    imgshape = z.shape
    if len(xi) != len(yi):
        raise ValueError('xi and yi must be equal-length!')
    if len(xi) < 2:
        raise ValueError('xi or yi must contain at least two elements!')
    for idx, ll in enumerate(xi):
        if not 0 < ll < imgshape[1]:
            raise ValueError('xi out of range!')
        if not 0 < yi[idx] < imgshape[0]:
            raise ValueError('yi out of range!')  # xx, yy = np.meshgrid(x, y)
    if len(xi) == 2:
        length = np.hypot(np.diff(xi), np.diff(yi))[0]
        x, y = np.linspace(xi[0], xi[1], length), np.linspace(yi[0], yi[1], length)
    else:
        x, y = xi, yi
    if interp == 'cubic':
        zi = scipy.ndimage.map_coordinates(z, np.vstack((x, y)))
    else:
        zi = z[np.floor(y).astype(np.int), np.floor(x).astype(np.int)]

    return zi
