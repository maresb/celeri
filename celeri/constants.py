import numpy as np

# Lazy initialization for pyproj to reduce startup time
_GEOID = None


def _get_geoid():
    """Lazily initialize the WGS84 geoid for geodetic calculations."""
    global _GEOID
    if _GEOID is None:
        import pyproj

        _GEOID = pyproj.Geod(ellps="WGS84")
    return _GEOID


class _LazyGeoid:
    """Lazy proxy for GEOID that delays pyproj import until first access."""

    def __getattr__(self, name):
        return getattr(_get_geoid(), name)

    def __call__(self, *args, **kwargs):
        return _get_geoid()(*args, **kwargs)


GEOID = _LazyGeoid()
KM2M = 1.0e3
M2MM = 1.0e3

# Defer RADIUS_EARTH calculation to avoid pyproj import at module load
_RADIUS_EARTH = None


def _get_radius_earth():
    global _RADIUS_EARTH
    if _RADIUS_EARTH is None:
        geoid = _get_geoid()
        _RADIUS_EARTH = np.float64((geoid.a + geoid.b) / 2)
    return _RADIUS_EARTH


class _LazyRadiusEarth:
    """Lazy proxy for RADIUS_EARTH that delays calculation."""

    def __float__(self):
        return float(_get_radius_earth())

    def __repr__(self):
        return repr(_get_radius_earth())

    def __mul__(self, other):
        return _get_radius_earth() * other

    def __rmul__(self, other):
        return other * _get_radius_earth()

    def __truediv__(self, other):
        return _get_radius_earth() / other

    def __rtruediv__(self, other):
        return other / _get_radius_earth()

    def __add__(self, other):
        return _get_radius_earth() + other

    def __radd__(self, other):
        return other + _get_radius_earth()

    def __sub__(self, other):
        return _get_radius_earth() - other

    def __rsub__(self, other):
        return other - _get_radius_earth()

    def __neg__(self):
        return -_get_radius_earth()

    def __pos__(self):
        return +_get_radius_earth()

    def __pow__(self, other):
        return _get_radius_earth() ** other

    def __eq__(self, other):
        return _get_radius_earth() == other

    def __ne__(self, other):
        return _get_radius_earth() != other

    def __lt__(self, other):
        return _get_radius_earth() < other

    def __le__(self, other):
        return _get_radius_earth() <= other

    def __gt__(self, other):
        return _get_radius_earth() > other

    def __ge__(self, other):
        return _get_radius_earth() >= other

    def __array__(self, dtype=None, copy=None):
        val = _get_radius_earth()
        if dtype is not None:
            return np.array(val, dtype=dtype)
        return np.array(val)


RADIUS_EARTH = _LazyRadiusEarth()
DEG_PER_MYR_TO_RAD_PER_YR = 1 / 1e3
# The conversion should be 1 / 1e3. Linear units for Cartesian conversions are
# in meters, but we need to convert them to mm to be consistent with mm/yr
# geodetic constraints units. Rotation constraints are expressed in deg/Myr,
# and when applied in celeri we are effectively using m*rad/Myr. To convert
# to the right rate units, we need 1e-3*m*rad/Myr. This conversion is applied in
# get_data_vector (JPL 12/31/23)
N_MESH_DIM = 3
EPS = np.finfo(float).eps
