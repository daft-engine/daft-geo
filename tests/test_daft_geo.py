from __future__ import annotations

import math

import pytest

import daft
from daft import col
from daft.session import Session


@pytest.fixture
def sess():
    """Session with daft_geo extension loaded."""
    import daft_geo

    s = Session()
    s.load_extension(daft_geo)
    return s


def test_point2d(sess):
    """Construct Point2D from two float columns."""
    from daft_geo import point2d

    df = daft.from_pydict({"lat": [37.7749, 34.0522], "lon": [-122.4194, -118.2437]})
    with sess:
        result = df.select(point2d(col("lat"), col("lon"))).collect()

    schema = result.schema()
    assert "geoarrow.point" in str(schema)


def test_point2d_null_propagation(sess):
    """Null inputs produce null points."""
    from daft_geo import point2d, x

    df = daft.from_pydict({"lat": [37.7749, None], "lon": [-122.4194, -118.2437]})
    with sess:
        result = (
            df.select(point2d(col("lat"), col("lon")).alias("pt"))
            .select(x(col("pt")))
            .collect()
            .to_pydict()
        )
    values = next(iter(result.values()))
    assert abs(values[0] - 37.7749) < 1e-6
    assert values[1] is None


def test_extract_x_y(sess):
    """Extract x and y coordinates from a Point2D."""
    from daft_geo import point2d, x, y

    df = daft.from_pydict({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    with sess:
        result = (
            df.with_column("pt", point2d(col("a"), col("b")))
            .select(x(col("pt")).alias("x_val"), y(col("pt")).alias("y_val"))
            .collect()
            .to_pydict()
        )
    assert result["x_val"] == [1.0, 2.0]
    assert result["y_val"] == [3.0, 4.0]


def test_euclidean_distance(sess):
    """Euclidean distance between 2D points."""
    from daft_geo import distance, point2d

    df = daft.from_pydict({"x1": [0.0], "y1": [0.0], "x2": [3.0], "y2": [4.0]})
    with sess:
        result = (
            df.with_column("p1", point2d(col("x1"), col("y1")))
            .with_column("p2", point2d(col("x2"), col("y2")))
            .select(distance(col("p1"), col("p2")))
            .collect()
            .to_pydict()
        )
    values = next(iter(result.values()))
    assert abs(values[0] - 5.0) < 1e-6


def test_haversine_sf_to_la(sess):
    """Haversine distance between SF and LA should be ~559 km."""
    from daft_geo import haversine, point2d

    df = daft.from_pydict(
        {
            "lat1": [37.7749],
            "lon1": [-122.4194],
            "lat2": [34.0522],
            "lon2": [-118.2437],
        }
    )
    with sess:
        result = (
            df.with_column("p1", point2d(col("lat1"), col("lon1")))
            .with_column("p2", point2d(col("lat2"), col("lon2")))
            .select(haversine(col("p1"), col("p2")))
            .collect()
            .to_pydict()
        )
    values = next(iter(result.values()))
    assert abs(values[0] - 559_120.0) < 500.0


def test_point3d_distance(sess):
    """3D Euclidean distance: (0,0,0) to (1,1,1) = sqrt(3)."""
    from daft_geo import distance, point3d

    df = daft.from_pydict(
        {
            "x1": [0.0],
            "y1": [0.0],
            "z1": [0.0],
            "x2": [1.0],
            "y2": [1.0],
            "z2": [1.0],
        }
    )
    with sess:
        result = (
            df.with_column("p1", point3d(col("x1"), col("y1"), col("z1")))
            .with_column("p2", point3d(col("x2"), col("y2"), col("z2")))
            .select(distance(col("p1"), col("p2")))
            .collect()
            .to_pydict()
        )
    values = next(iter(result.values()))
    assert abs(values[0] - math.sqrt(3.0)) < 1e-6


def test_point2d_dtype(sess):
    """Point2D columns have the daft_geo.point2d extension type."""
    from daft_geo import Point2D, point2d

    df = daft.from_pydict({"a": [1.0], "b": [2.0]})
    with sess:
        result = df.select(point2d(col("a"), col("b")).alias("pt")).collect()

    assert result.schema()["pt"].dtype == Point2D


def test_point3d_dtype(sess):
    """Point3D columns have the daft_geo.point3d extension type."""
    from daft_geo import Point3D, point3d

    df = daft.from_pydict({"a": [1.0], "b": [2.0], "c": [3.0]})
    with sess:
        result = df.select(point3d(col("a"), col("b"), col("c")).alias("pt")).collect()

    assert result.schema()["pt"].dtype == Point3D


def test_function_not_available_without_extension():
    """Functions should not be available without loading the extension."""
    from daft_geo import point2d

    sess = Session()
    with sess:
        df = daft.from_pydict({"a": [1.0], "b": [2.0]})
        with pytest.raises(Exception, match="not found"):
            df.select(point2d(col("a"), col("b"))).collect()
