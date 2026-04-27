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
        if !GeoPointArray::matches_schema(&args[0]) {
            return Err(DaftError::TypeError(
                "geo_x: argument must be a geoarrow.point".into(),
            ));
        }
        let input = Field::try_from(&args[0]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let out = Field::new(input.name(), DataType::Float64, true);
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
        if !GeoPointArray::matches_schema(&args[0]) {
            return Err(DaftError::TypeError(
                "geo_y: argument must be a geoarrow.point".into(),
            ));
        }
        let input = Field::try_from(&args[0]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let out = Field::new(input.name(), DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        extract_component(args, 1, "geo_y")
    }
}

fn extract_component(args: Vec<ArrowData>, index: usize, fn_name: &str) -> DaftResult<ArrowData> {
    if args.len() != 1 {
        return Err(DaftError::RuntimeError(format!(
            "{fn_name}: expected 1 argument, got {}",
            args.len()
        )));
    }
    let arg = args.into_iter().next().unwrap();
    let input_field =
        Field::try_from(&arg.schema).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
    let output_name = input_field.name().clone();
    let pts = GeoPointArray::from_ffi(arg)?;
    let len = pts.len();
    let mut values = Vec::with_capacity(len);
    let mut null_count = 0;
    let mut validity = BooleanBufferBuilder::new(len);

    for row in 0..len {
        match pts
            .get(row)
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?
        {
            Some(pt) => {
                let coord = pt.coord().ok_or_else(|| {
                    DaftError::RuntimeError(format!("{fn_name}: empty point at row {row}"))
                })?;
                values.push(coord.nth_or_panic(index));
                validity.append(true);
            }
            None => {
                values.push(0.0);
                validity.append(false);
                null_count += 1;
            }
        }
    }

    let result: ArrayRef = if null_count > 0 {
        let nulls = NullBuffer::from(validity.finish());
        Arc::new(Float64Array::new(values.into(), Some(nulls)))
    } else {
        Arc::new(Float64Array::from(values))
    };

    let out_field = Field::new(&output_name, DataType::Float64, null_count > 0);
    export_arrow(&out_field, result)
}
