use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanBufferBuilder, Float64Array},
    buffer::NullBuffer,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;
use geo_traits::{CoordTrait, PointTrait};
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor};

use crate::types::{GeoArrowFfi, GeoPointArray, export_arrow};

const EARTH_RADIUS_M: f64 = 6_371_000.0;

pub struct GeoDistance;

impl DaftScalarFunction for GeoDistance {
    fn name(&self) -> &CStr {
        c"geo_distance"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 2 {
            return Err(DaftError::TypeError(format!(
                "geo_distance: expected 2 arguments, got {}",
                args.len()
            )));
        }
        if !GeoPointArray::matches_schema(&args[0]) || !GeoPointArray::matches_schema(&args[1]) {
            return Err(DaftError::TypeError(
                "geo_distance: both arguments must be geoarrow.point".into(),
            ));
        }
        let d0 = GeoPointArray::dimension_of_schema(&args[0])?;
        let d1 = GeoPointArray::dimension_of_schema(&args[1])?;
        if d0 != d1 {
            return Err(DaftError::TypeError(format!(
                "geo_distance: dimension mismatch: {d0:?} vs {d1:?}"
            )));
        }
        let out = Field::new("geo_distance", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        if args.len() != 2 {
            return Err(DaftError::RuntimeError(format!(
                "geo_distance: expected 2 arguments, got {}",
                args.len()
            )));
        }
        let mut iter = args.into_iter();
        let a = GeoPointArray::from_ffi(iter.next().unwrap())?;
        let b = GeoPointArray::from_ffi(iter.next().unwrap())?;

        let len = a.len();
        let mut results = Vec::with_capacity(len);
        let mut null_count = 0;
        let mut validity = BooleanBufferBuilder::new(len);

        for row in 0..len {
            let pa = a.get(row).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
            let pb = b.get(row).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
            match (pa.and_then(|p| p.coord()), pb.and_then(|p| p.coord())) {
                (Some(ca), Some(cb)) => {
                    let dims = ca.dim().size();
                    let sum_sq: f64 = (0..dims)
                        .map(|d| (ca.nth_or_panic(d) - cb.nth_or_panic(d)).powi(2))
                        .sum();
                    results.push(sum_sq.sqrt());
                    validity.append(true);
                }
                _ => {
                    results.push(0.0);
                    validity.append(false);
                    null_count += 1;
                }
            }
        }

        let result: ArrayRef = if null_count > 0 {
            let nulls = NullBuffer::from(validity.finish());
            Arc::new(Float64Array::new(results.into(), Some(nulls)))
        } else {
            Arc::new(Float64Array::from(results))
        };

        let out_field = Field::new("geo_distance", DataType::Float64, null_count > 0);
        export_arrow(&out_field, result)
    }
}

pub struct GeoHaversine;

impl DaftScalarFunction for GeoHaversine {
    fn name(&self) -> &CStr {
        c"geo_haversine"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 2 {
            return Err(DaftError::TypeError(format!(
                "geo_haversine: expected 2 arguments, got {}",
                args.len()
            )));
        }
        for (i, arg) in args.iter().enumerate() {
            if !GeoPointArray::matches_schema(arg) {
                return Err(DaftError::TypeError(format!(
                    "geo_haversine: argument {i} must be a geoarrow.point"
                )));
            }
            let dim = GeoPointArray::dimension_of_schema(arg)?;
            if dim != geoarrow_schema::Dimension::XY {
                return Err(DaftError::TypeError(
                    "geo_haversine: arguments must be 2D points (lat, lon)".into(),
                ));
            }
        }
        let out = Field::new("geo_haversine", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        if args.len() != 2 {
            return Err(DaftError::RuntimeError(format!(
                "geo_haversine: expected 2 arguments, got {}",
                args.len()
            )));
        }
        let mut iter = args.into_iter();
        let a = GeoPointArray::from_ffi(iter.next().unwrap())?;
        let b = GeoPointArray::from_ffi(iter.next().unwrap())?;

        let len = a.len();
        let mut results = Vec::with_capacity(len);
        let mut null_count = 0;
        let mut validity = BooleanBufferBuilder::new(len);

        for row in 0..len {
            let pa = a.get(row).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
            let pb = b.get(row).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
            match (pa.and_then(|p| p.coord()), pb.and_then(|p| p.coord())) {
                (Some(ca), Some(cb)) => {
                    let (lat1, lon1) = (ca.x(), ca.y());
                    let (lat2, lon2) = (cb.x(), cb.y());
                    match haversine_meters(lat1, lon1, lat2, lon2) {
                        Some(d) => {
                            results.push(d);
                            validity.append(true);
                        }
                        None => {
                            results.push(0.0);
                            validity.append(false);
                            null_count += 1;
                        }
                    }
                }
                _ => {
                    results.push(0.0);
                    validity.append(false);
                    null_count += 1;
                }
            }
        }

        let result: ArrayRef = if null_count > 0 {
            let nulls = NullBuffer::from(validity.finish());
            Arc::new(Float64Array::new(results.into(), Some(nulls)))
        } else {
            Arc::new(Float64Array::from(results))
        };

        let out_field = Field::new("geo_haversine", DataType::Float64, null_count > 0);
        export_arrow(&out_field, result)
    }
}

fn haversine_meters(lat1: f64, lon1: f64, lat2: f64, lon2: f64) -> Option<f64> {
    if !is_valid_lat_lon(lat1, lon1) || !is_valid_lat_lon(lat2, lon2) {
        return None;
    }
    let (lat1_r, lon1_r) = (lat1.to_radians(), lon1.to_radians());
    let (lat2_r, lon2_r) = (lat2.to_radians(), lon2.to_radians());
    let dlat = lat2_r - lat1_r;
    let dlon = lon2_r - lon1_r;
    let a = (dlat * 0.5).sin().powi(2) + lat1_r.cos() * lat2_r.cos() * (dlon * 0.5).sin().powi(2);
    Some(EARTH_RADIUS_M * 2.0 * a.clamp(0.0, 1.0).sqrt().asin())
}

fn is_valid_lat_lon(lat: f64, lon: f64) -> bool {
    lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-180.0..=180.0).contains(&lon)
}
