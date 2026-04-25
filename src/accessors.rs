use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, BooleanBufferBuilder, Float64Array, cast::AsArray, types::Float64Type,
    },
    buffer::NullBuffer,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;

use crate::types::{from_array, is_point, to_array};

fn extract_component(args: Vec<ArrowData>, index: usize, fn_name: &str) -> DaftResult<ArrowData> {
    if args.len() != 1 {
        return Err(DaftError::RuntimeError(format!(
            "{fn_name}: expected 1 argument, got {}",
            args.len()
        )));
    }
    let (_, array) = to_array(args.into_iter().next().unwrap())?;
    let fsl = array
        .as_fixed_size_list_opt()
        .ok_or_else(|| DaftError::TypeError(format!("{fn_name}: expected FixedSizeList input")))?;

    let dims = fsl.value_length() as usize;
    if index >= dims {
        return Err(DaftError::RuntimeError(format!(
            "{fn_name}: index {index} out of bounds for {dims}D point"
        )));
    }

    let inner_values = fsl.values().as_primitive::<Float64Type>();
    let len = fsl.len();
    let mut values = Vec::with_capacity(len);
    let mut null_count = 0;
    let mut validity = BooleanBufferBuilder::new(len);

    for row in 0..len {
        if fsl.is_null(row) {
            values.push(0.0);
            validity.append(false);
            null_count += 1;
        } else {
            values.push(inner_values.value(row * dims + index));
            validity.append(true);
        }
    }

    let result: ArrayRef = if null_count > 0 {
        let nulls = NullBuffer::from(validity.finish());
        Arc::new(Float64Array::new(values.into(), Some(nulls)))
    } else {
        Arc::new(Float64Array::from(values))
    };

    let out_field = Field::new(fn_name, DataType::Float64, null_count > 0);
    from_array(&out_field, result)
}

pub struct GeoX;

impl DaftScalarFunction for GeoX {
    fn name(&self) -> &CStr {
        c"geo_x"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 1 {
            return Err(DaftError::TypeError(format!(
                "geo_x: expected 1 argument, got {}",
                args.len()
            )));
        }
        if !is_point(&args[0]) {
            return Err(DaftError::TypeError(
                "geo_x: argument must be a Point2D or Point3D".into(),
            ));
        }
        let out = Field::new("geo_x", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        extract_component(args, 0, "geo_x")
    }
}

pub struct GeoY;

impl DaftScalarFunction for GeoY {
    fn name(&self) -> &CStr {
        c"geo_y"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 1 {
            return Err(DaftError::TypeError(format!(
                "geo_y: expected 1 argument, got {}",
                args.len()
            )));
        }
        if !is_point(&args[0]) {
            return Err(DaftError::TypeError(
                "geo_y: argument must be a Point2D or Point3D".into(),
            ));
        }
        let out = Field::new("geo_y", DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        extract_component(args, 1, "geo_y")
    }
}
