use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{ArrayRef, BooleanBufferBuilder, Float64Array},
    buffer::NullBuffer,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;
use geo_traits::{CoordTrait, PointTrait};
use geoarrow_array::{GeoArrowArray, GeoArrowArrayAccessor};

use crate::types::{PointArg, PointFieldArg, export_arrow};

pub struct GeoX;

impl DaftScalarFunction for GeoX {
    fn name(&self) -> &CStr {
        c"geo_x"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        let args = PointFieldArg::try_from(args)?;
        let out = Field::new(args.field.name(), DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        extract_component(args, 0)
    }
}

pub struct GeoY;

impl DaftScalarFunction for GeoY {
    fn name(&self) -> &CStr {
        c"geo_y"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        let args = PointFieldArg::try_from(args)?;
        let out = Field::new(args.field.name(), DataType::Float64, true);
        ArrowSchema::try_from(&out).map_err(|e| DaftError::TypeError(e.to_string()))
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        extract_component(args, 1)
    }
}

fn extract_component(args: Vec<ArrowData>, index: usize) -> DaftResult<ArrowData> {
    let PointArg { name, points } = args.try_into()?;
    let len = points.len();
    let mut values = Vec::with_capacity(len);
    let mut null_count = 0;
    let mut validity = BooleanBufferBuilder::new(len);

    for row in 0..len {
        match points
            .get(row)
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?
        {
            Some(pt) => {
                let coord = pt
                    .coord()
                    .ok_or_else(|| DaftError::RuntimeError(format!("empty point at row {row}")))?;
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

    let out_field = Field::new(&name, DataType::Float64, null_count > 0);
    export_arrow(&out_field, result)
}
