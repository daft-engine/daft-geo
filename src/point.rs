use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{Array, ArrayRef, Float64Array, cast::AsArray, types::Float64Type},
    buffer::{NullBuffer, ScalarBuffer},
    compute::cast,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;
use geoarrow_array::array::{InterleavedCoordBuffer, PointArray};
use geoarrow_schema::{Dimension, Metadata};

use crate::types::{GeoPointArray, import_arrow};

fn validate_numeric(field: &Field, arg_name: &str) -> DaftResult<()> {
    match field.data_type() {
        DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(()),
        other => Err(DaftError::TypeError(format!(
            "{arg_name}: expected numeric type, got {other:?}"
        ))),
    }
}

fn cast_to_f64(array: &ArrayRef) -> DaftResult<Float64Array> {
    if *array.data_type() == DataType::Float64 {
        return Ok(array.as_primitive::<Float64Type>().clone());
    }
    Ok(cast(array, &DataType::Float64)
        .map_err(|e| DaftError::RuntimeError(e.to_string()))?
        .as_primitive::<Float64Type>()
        .clone())
}

fn build_point_array(
    arrays: Vec<ArrayRef>,
    dim: Dimension,
    output_name: &str,
) -> DaftResult<ArrowData> {
    let len = arrays[0].len();
    let f64_arrays: Vec<Float64Array> = arrays
        .iter()
        .map(cast_to_f64)
        .collect::<DaftResult<Vec<_>>>()?;

    let mut coords = Vec::with_capacity(len * dim.size());
    for row in 0..len {
        for arr in &f64_arrays {
            coords.push(arr.value(row));
        }
    }

    let coord_buffer = InterleavedCoordBuffer::new(ScalarBuffer::from(coords), dim);

    let validity = f64_arrays
        .iter()
        .filter_map(|a| a.nulls())
        .map(|n| n.inner().clone())
        .reduce(|a, b| &a & &b)
        .map(NullBuffer::new);

    let point_array = PointArray::new(coord_buffer.into(), validity, Arc::new(Metadata::default()));
    GeoPointArray(point_array).into_ffi(output_name)
}

pub struct GeoPoint2D;

impl DaftScalarFunction for GeoPoint2D {
    fn name(&self) -> &CStr {
        c"geo_point2d"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 2 {
            return Err(DaftError::TypeError(format!(
                "geo_point2d: expected 2 arguments, got {}",
                args.len()
            )));
        }
        let f0 = Field::try_from(&args[0]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let f1 = Field::try_from(&args[1]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        validate_numeric(&f0, "x")?;
        validate_numeric(&f1, "y")?;
        GeoPointArray::output_schema(f0.name(), Dimension::XY)
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        if args.len() != 2 {
            return Err(DaftError::RuntimeError(format!(
                "geo_point2d: expected 2 arguments, got {}",
                args.len()
            )));
        }
        let output_name = Field::try_from(&args[0].schema)
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?
            .name()
            .clone();
        let arrays: Vec<ArrayRef> = args
            .into_iter()
            .map(|d| import_arrow(d).map(|(_, arr)| arr))
            .collect::<DaftResult<Vec<_>>>()?;

        build_point_array(arrays, Dimension::XY, &output_name)
    }
}

pub struct GeoPoint3D;

impl DaftScalarFunction for GeoPoint3D {
    fn name(&self) -> &CStr {
        c"geo_point3d"
    }

    fn return_field(&self, args: &[ArrowSchema]) -> DaftResult<ArrowSchema> {
        if args.len() != 3 {
            return Err(DaftError::TypeError(format!(
                "geo_point3d: expected 3 arguments, got {}",
                args.len()
            )));
        }
        let f0 = Field::try_from(&args[0]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let f1 = Field::try_from(&args[1]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let f2 = Field::try_from(&args[2]).map_err(|e| DaftError::TypeError(e.to_string()))?;
        validate_numeric(&f0, "x")?;
        validate_numeric(&f1, "y")?;
        validate_numeric(&f2, "z")?;
        GeoPointArray::output_schema(f0.name(), Dimension::XYZ)
    }

    fn call(&self, args: Vec<ArrowData>) -> DaftResult<ArrowData> {
        if args.len() != 3 {
            return Err(DaftError::RuntimeError(format!(
                "geo_point3d: expected 3 arguments, got {}",
                args.len()
            )));
        }
        let output_name = Field::try_from(&args[0].schema)
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?
            .name()
            .clone();
        let arrays: Vec<ArrayRef> = args
            .into_iter()
            .map(|d| import_arrow(d).map(|(_, arr)| arr))
            .collect::<DaftResult<Vec<_>>>()?;

        build_point_array(arrays, Dimension::XYZ, &output_name)
    }
}
