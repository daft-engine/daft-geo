use std::{collections::HashMap, sync::Arc};

use arrow::{
    array::{ArrayRef, make_array},
    datatypes::{DataType, Field},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi},
};
use daft_ext::prelude::{ArrowData, ArrowSchema, DaftError, DaftResult};

pub const POINT2D_EXT_NAME: &str = "daft_geo.point2d";
pub const POINT3D_EXT_NAME: &str = "daft_geo.point3d";

const EXT_NAME_KEY: &str = "ARROW:extension:name";

pub fn point2d_field(name: &str) -> Field {
    let inner = Field::new("item", DataType::Float64, false);
    Field::new(name, DataType::FixedSizeList(Arc::new(inner), 2), true).with_metadata(
        HashMap::from([(EXT_NAME_KEY.into(), POINT2D_EXT_NAME.into())]),
    )
}

pub fn point3d_field(name: &str) -> Field {
    let inner = Field::new("item", DataType::Float64, false);
    Field::new(name, DataType::FixedSizeList(Arc::new(inner), 3), true).with_metadata(
        HashMap::from([(EXT_NAME_KEY.into(), POINT3D_EXT_NAME.into())]),
    )
}

pub fn extension_name(schema: &ArrowSchema) -> Option<String> {
    let field = Field::try_from(schema).ok()?;
    field.metadata().get(EXT_NAME_KEY).cloned()
}

pub fn is_point2d(schema: &ArrowSchema) -> bool {
    extension_name(schema).as_deref() == Some(POINT2D_EXT_NAME)
}

pub fn is_point3d(schema: &ArrowSchema) -> bool {
    extension_name(schema).as_deref() == Some(POINT3D_EXT_NAME)
}

pub fn is_point(schema: &ArrowSchema) -> bool {
    is_point2d(schema) || is_point3d(schema)
}

pub fn point_dims(schema: &ArrowSchema) -> DaftResult<i32> {
    let field = Field::try_from(schema).map_err(|e| DaftError::TypeError(e.to_string()))?;
    match field.data_type() {
        DataType::FixedSizeList(_, dims) => Ok(*dims),
        other => Err(DaftError::TypeError(format!(
            "expected FixedSizeList, got {other:?}"
        ))),
    }
}

pub fn to_array(data: ArrowData) -> DaftResult<(Field, ArrayRef)> {
    let field = Field::try_from(&data.schema).map_err(|e| DaftError::TypeError(e.to_string()))?;
    let ffi_schema: FFI_ArrowSchema = data.schema.into();
    let ffi_array: FFI_ArrowArray = data.array.into();
    let array_data = unsafe { from_ffi(ffi_array, &ffi_schema) }
        .map_err(|e| DaftError::RuntimeError(e.to_string()))?;
    Ok((field, make_array(array_data)))
}

pub fn from_array(field: &Field, array: ArrayRef) -> DaftResult<ArrowData> {
    let schema = ArrowSchema::try_from(field).map_err(|e| DaftError::TypeError(e.to_string()))?;
    let (ffi_array, _ffi_schema) =
        to_ffi(&array.to_data()).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
    Ok(ArrowData {
        schema,
        array: ffi_array.into(),
    })
}
