use std::{ffi::CStr, sync::Arc};

use arrow::{
    array::{
        Array, ArrayRef, BooleanBufferBuilder, FixedSizeListArray, Float64Array, cast::AsArray,
        types::Float64Type,
    },
    buffer::NullBuffer,
    compute::cast,
    datatypes::{DataType, Field},
};
use daft_ext::prelude::*;
use geoarrow_schema::Dimension;

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
        return Ok(array.clone().as_primitive().clone());
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
    let dims = dim.size() as i32;
    let f64_arrays: Vec<Float64Array> = arrays
        .iter()
        .map(cast_to_f64)
        .collect::<DaftResult<Vec<_>>>()?;

    let mut values = Vec::with_capacity(len * dims as usize);
    let mut null_count = 0;
    let mut validity = BooleanBufferBuilder::new(len);

    for row in 0..len {
        let any_null = f64_arrays.iter().any(|a| a.is_null(row));
        if any_null {
            values.extend(std::iter::repeat_n(0.0, dims as usize));
            validity.append(false);
            null_count += 1;
        } else {
            for arr in &f64_arrays {
                values.push(arr.value(row));
            }
            validity.append(true);
        }
    }

    let values_array = Arc::new(Float64Array::from(values)) as ArrayRef;
    let inner_field = Arc::new(Field::new("xy", DataType::Float64, false));
    let nulls = if null_count > 0 {
        Some(NullBuffer::from(validity.finish()))
    } else {
        None
    };
    let fsl = FixedSizeListArray::try_new(inner_field, dims, values_array, nulls)
        .map_err(|e| DaftError::RuntimeError(e.to_string()))?;

    GeoPointArray::from_fixed_size_list(fsl, dim)?.into_ffi(output_name)
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
