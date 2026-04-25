use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, BooleanBufferBuilder, Float64Array, cast::AsArray, types::Float64Type,
    },
    buffer::NullBuffer,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;

use crate::types::{from_array, is_point, is_point2d, point_dims, to_array};

const EARTH_RADIUS_M: f64 = 6_371_000.0;

fn pairwise_distance(
    args: Vec<ArrowData>,
    fn_name: &str,
    f: impl Fn(&[f64], &[f64]) -> Option<f64>,
) -> DaftResult<ArrowData> {
    if args.len() != 2 {
        return Err(DaftError::RuntimeError(format!(
            "{fn_name}: expected 2 arguments, got {}",
            args.len()
        )));
    }
    let mut iter = args.into_iter();
    let (_, a) = to_array(iter.next().unwrap())?;
    let (_, b) = to_array(iter.next().unwrap())?;

    let fsl_a = a
        .as_fixed_size_list_opt()
        .ok_or_else(|| DaftError::TypeError(format!("{fn_name}: arg 0 must be a point")))?;
    let fsl_b = b
        .as_fixed_size_list_opt()
        .ok_or_else(|| DaftError::TypeError(format!("{fn_name}: arg 1 must be a point")))?;

    let dims = fsl_a.value_length() as usize;
    if fsl_b.value_length() as usize != dims {
        return Err(DaftError::TypeError(format!(
            "{fn_name}: dimension mismatch: {} vs {}",
            dims,
            fsl_b.value_length()
        )));
    }

    let len = fsl_a.len();
    if fsl_b.len() != len {
        return Err(DaftError::RuntimeError(format!(
            "{fn_name}: length mismatch: {} vs {}",
            len,
            fsl_b.len()
        )));
    }

    let vals_a = fsl_a.values().as_primitive::<Float64Type>();
    let vals_b = fsl_b.values().as_primitive::<Float64Type>();

    let mut results = Vec::with_capacity(len);
    let mut null_count = 0;
    let mut validity = BooleanBufferBuilder::new(len);

    for row in 0..len {
        if fsl_a.is_null(row) || fsl_b.is_null(row) {
            results.push(0.0);
            validity.append(false);
            null_count += 1;
            continue;
        }

        let offset_a = row * dims;
        let offset_b = row * dims;
        let pa: Vec<f64> = (0..dims).map(|d| vals_a.value(offset_a + d)).collect();
        let pb: Vec<f64> = (0..dims).map(|d| vals_b.value(offset_b + d)).collect();

        match f(&pa, &pb) {
            Some(v) => {
                results.push(v);
                validity.append(true);
            }
            None => {
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

    let out_field = Field::new(fn_name, DataType::Float64, null_count > 0);
    from_array(&out_field, result)
}

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
        if !is_point(&args[0]) || !is_point(&args[1]) {
            return Err(DaftError::TypeError(
                "geo_distance: both arguments must be points".into(),
            ));
        }
        let d0 = point_dims(&args[0])?;
        let d1 = point_dims(&args[1])?;
        if d0 != d1 {
            return Err(DaftError::TypeError(format!(
                "geo_distance: dimension mismatch: {d0} vs {d1}"
            )));
        }
        let out = Field::new("geo_distance", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        pairwise_distance(args, "geo_distance", |a, b| {
            let sum_sq: f64 = a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum();
            Some(sum_sq.sqrt())
        })
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
        if !is_point2d(&args[0]) || !is_point2d(&args[1]) {
            return Err(DaftError::TypeError(
                "geo_haversine: both arguments must be Point2D (lat, lon)".into(),
            ));
        }
        let out = Field::new("geo_haversine", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        pairwise_distance(args, "geo_haversine", |a, b| {
            let (lat1, lon1) = (a[0], a[1]);
            let (lat2, lon2) = (b[0], b[1]);

            if !is_valid_lat_lon(lat1, lon1) || !is_valid_lat_lon(lat2, lon2) {
                return None;
            }

            let lat1_r = lat1.to_radians();
            let lon1_r = lon1.to_radians();
            let lat2_r = lat2.to_radians();
            let lon2_r = lon2.to_radians();

            let dlat = lat2_r - lat1_r;
            let dlon = lon2_r - lon1_r;

            let a = (dlat * 0.5).sin().powi(2)
                + lat1_r.cos() * lat2_r.cos() * (dlon * 0.5).sin().powi(2);
            let a = a.clamp(0.0, 1.0);

            Some(EARTH_RADIUS_M * 2.0 * a.sqrt().asin())
        })
    }
}

fn is_valid_lat_lon(lat: f64, lon: f64) -> bool {
    lat.is_finite()
        && lon.is_finite()
        && (-90.0..=90.0).contains(&lat)
        && (-180.0..=180.0).contains(&lon)
}
