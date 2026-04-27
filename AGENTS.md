# daft-geo

Standalone geospatial extension package for [Daft](https://github.com/Eventual-Inc/Daft), targeting physical AI workflows.

## What this is

A native Rust extension that plugs into Daft via the `daft-ext` C ABI. It provides custom geo datatypes (as Arrow extension types) and scalar functions operating on them. The package is fully standalone — it depends only on `daft-ext` from crates.io and `arrow` v57.

## Architecture

```
daft-geo/
├── Cargo.toml              # Rust cdylib, depends on daft-ext + arrow
├── pyproject.toml           # Python package (setuptools + setuptools-rust)
├── setup.py                 # RustExtension binding (Binding.NoBinding)
├── daft_geo/
│   └── __init__.py          # Python API: types (Point2D, Point3D) + functions
├── src/
│   ├── lib.rs               # #[daft_extension] entry point, registers all functions
│   ├── types.rs             # Extension type constants, Arrow FFI helpers (to_array/from_array)
│   ├── point.rs             # geo_point2d, geo_point3d constructors
│   ├── distance.rs          # geo_distance (Euclidean), geo_haversine (great-circle)
│   └── accessors.rs         # geo_x, geo_y coordinate extractors
└── tests/
    └── test_daft_geo.py     # pytest suite (9 tests)
```

## Key concepts

### Extension types
Types are Arrow extension types — convention-based, encoded as metadata on `FixedSizeList<Float64>`:
- **Point2D**: `FixedSizeList[Float64, 2]` with `ARROW:extension:name = "daft_geo.point2d"`
- **Point3D**: `FixedSizeList[Float64, 3]` with `ARROW:extension:name = "daft_geo.point3d"`

On the Python side, `daft_geo.Point2D` and `daft_geo.Point3D` are `DataType.extension(...)` objects that can be used for casting and schema comparison.

### How functions work
Each function implements `DaftScalarFunction` from the `daft-ext` SDK:
- `name()` → C string function name
- `return_field(args: &[ArrowSchema])` → validates input types, returns output schema
- `call(args: Vec<ArrowData>)` → receives Arrow C Data Interface arrays, computes, returns result

The `types.rs` module provides `to_array()` and `from_array()` helpers that convert between daft-ext's `ArrowData`/`ArrowSchema` and arrow-rs `ArrayRef`/`Field` types via FFI.

### How the extension is loaded
1. `setuptools-rust` compiles the Rust code and places the cdylib inside the `daft_geo/` Python package
2. User calls `session.load_extension(daft_geo)` which finds the `.so`/`.dylib` and calls `daft_module_magic()` via dlopen
3. The `install()` function registers all 6 scalar functions with the session
4. Functions are then callable via `daft.get_function("geo_point2d", ...)` or the Python wrappers

## Dev workflow

```bash
# Install in dev mode (builds Rust + installs Python package)
uv sync --extra test

# Or build Rust only
cargo build --release

# Run tests
DAFT_RUNNER=native python -m pytest tests/ -v
```

## Current functions

| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `geo_point2d` | x: Float64, y: Float64 | Point2D | Construct 2D point |
| `geo_point3d` | x: Float64, y: Float64, z: Float64 | Point3D | Construct 3D point |
| `geo_distance` | p1: Point2D/3D, p2: Point2D/3D | Float64 | Euclidean distance |
| `geo_haversine` | p1: Point2D, p2: Point2D | Float64 | Great-circle distance (lat/lon degrees → meters) |
| `geo_x` | p: Point2D/3D | Float64 | Extract x coordinate |
| `geo_y` | p: Point2D/3D | Float64 | Extract y coordinate |

## Usage

```python
import daft
from daft import col
from daft.session import Session
import daft_geo

sess = Session()
sess.load_extension(daft_geo)

df = daft.from_pydict({"lat": [37.7749, 34.0522], "lon": [-122.4194, -118.2437]})
with sess:
    df = df.with_column("point", daft_geo.point2d(col("lat"), col("lon")))
    df = df.with_column("x", daft_geo.x(col("point")))
    df.show()
```

Users can also cast directly using the extension types:
```python
Point2D = daft_geo.Point2D
df = df.with_column("point", col("coords").cast(Point2D))
```

## What to work on next

- More geo types: `LineString`, `Polygon`, `BoundingBox2D`
- More functions: `geo_z` accessor, `geo_centroid`, `geo_contains`, `geo_intersects`, `geo_buffer`
- GeoArrow compatibility: align extension type names with the GeoArrow spec
- WKB/WKT parsing: `geo_from_wkb`, `geo_from_wkt`, `geo_to_wkt`
- Spatial indexing support (R-tree, H3)
- Benchmark against geopandas/sedona for common operations
