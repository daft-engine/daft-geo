use std::sync::Arc;

use daft_ext::{daft_extension, prelude::*};

mod accessors;
mod distance;
mod point;
mod types;

use accessors::{GeoX, GeoY};
use distance::{GeoDistance, GeoHaversine};
use point::{GeoPoint2D, GeoPoint3D};

#[daft_extension]
struct GeoExtension;

impl DaftExtension for GeoExtension {
    fn install(session: &mut dyn DaftSession) {
        session.define_function(Arc::new(GeoPoint2D));
        session.define_function(Arc::new(GeoPoint3D));
        session.define_function(Arc::new(GeoDistance));
        session.define_function(Arc::new(GeoHaversine));
        session.define_function(Arc::new(GeoX));
        session.define_function(Arc::new(GeoY));
    }
}
