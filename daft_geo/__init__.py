from __future__ import annotations

from typing import TYPE_CHECKING

import daft
from daft import DataType

if TYPE_CHECKING:
    from daft.expressions import Expression

# ---------------------------------------------------------------------------
# Types (GeoArrow-standard extension names)
# ---------------------------------------------------------------------------

Point2D = DataType.extension(
    "geoarrow.point",
    DataType.fixed_size_list(DataType.float64(), 2),
)

Point3D = DataType.extension(
    "geoarrow.point",
    DataType.fixed_size_list(DataType.float64(), 3),
)

# ---------------------------------------------------------------------------
# Functions
# ---------------------------------------------------------------------------


def point2d(x: Expression, y: Expression) -> Expression:
    """Construct a 2D point from x and y coordinates."""
    return daft.get_function("geo_point2d", x, y)


def point3d(x: Expression, y: Expression, z: Expression) -> Expression:
    """Construct a 3D point from x, y, and z coordinates."""
    return daft.get_function("geo_point3d", x, y, z)


def distance(p1: Expression, p2: Expression) -> Expression:
    """Compute Euclidean distance between two points."""
    return daft.get_function("geo_distance", p1, p2)


def haversine(p1: Expression, p2: Expression) -> Expression:
    """Compute great-circle distance in meters between two 2D points (lat, lon in degrees)."""
    return daft.get_function("geo_haversine", p1, p2)


def x(point: Expression) -> Expression:
    """Extract the x coordinate from a point."""
    return daft.get_function("geo_x", point)


def y(point: Expression) -> Expression:
    """Extract the y coordinate from a point."""
    return daft.get_function("geo_y", point)
