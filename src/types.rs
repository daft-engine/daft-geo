use std::sync::Arc;

use arrow::{
    array::{ArrayRef, FixedSizeListArray, make_array},
    datatypes::{DataType, Field},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi, to_ffi},
};
use daft_ext::prelude::{ArrowData, ArrowSchema, DaftError, DaftResult};
use geoarrow_array::{IntoArrow, array::PointArray};
use geoarrow_schema::{CoordType, Dimension, Metadata, PointType};

pub trait GeoArrowFfi: Sized {
    fn from_ffi(data: ArrowData) -> DaftResult<Self>;
    fn into_ffi(self, name: &str) -> DaftResult<ArrowData>;
    fn matches_schema(schema: &ArrowSchema) -> bool;
}

pub struct GeoPointArray(pub PointArray);

impl GeoArrowFfi for GeoPointArray {
    fn from_ffi(data: ArrowData) -> DaftResult<Self> {
        let (field, array) = import_arrow(data)?;
        let dim = Self::dimension_of_field(&field)?;
        let pa = PointArray::try_from((array.as_ref(), Self::geo_type(dim)))
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?;
        Ok(Self(pa))
    }

    fn into_ffi(self, name: &str) -> DaftResult<ArrowData> {
        let field = self.0.extension_type().to_field(name, true);
        export_arrow(&field, self.0.into_arrow())
    }

    fn matches_schema(schema: &ArrowSchema) -> bool {
        Self::dimension_of_schema(schema).is_ok()
    }
}

impl GeoPointArray {
    pub fn from_fixed_size_list(fsl: FixedSizeListArray, dim: Dimension) -> DaftResult<Self> {
        let arr: ArrayRef = Arc::new(fsl);
        let pa = PointArray::try_from((arr.as_ref(), Self::geo_type(dim)))
            .map_err(|e| DaftError::RuntimeError(e.to_string()))?;
        Ok(Self(pa))
    }

    pub fn dimension_of_schema(schema: &ArrowSchema) -> DaftResult<Dimension> {
        let field = Field::try_from(schema).map_err(|e| DaftError::TypeError(e.to_string()))?;
        let ext_name = field
            .metadata()
            .get("ARROW:extension:name")
            .map(|s| s.as_str());
        if ext_name != Some("geoarrow.point") {
            return Err(DaftError::TypeError(format!(
                "expected geoarrow.point extension, got {:?}",
                ext_name
            )));
        }
        Self::dimension_of_field(&field)
    }

    pub fn output_field(name: &str, dim: Dimension) -> Field {
        Self::geo_type(dim).to_field(name, true)
    }

    pub fn output_schema(name: &str, dim: Dimension) -> DaftResult<ArrowSchema> {
        ArrowSchema::try_from(&Self::output_field(name, dim))
            .map_err(|e| DaftError::TypeError(e.to_string()))
    }

    pub fn geo_type(dim: Dimension) -> PointType {
        PointType::new(dim, Arc::new(Metadata::default())).with_coord_type(CoordType::Interleaved)
    }

    fn dimension_of_field(field: &Field) -> DaftResult<Dimension> {
        match field.data_type() {
            DataType::FixedSizeList(_, 2) => Ok(Dimension::XY),
            DataType::FixedSizeList(_, 3) => Ok(Dimension::XYZ),
            DataType::Struct(fields) if fields.len() == 2 => Ok(Dimension::XY),
            DataType::Struct(fields) if fields.len() == 3 => Ok(Dimension::XYZ),
            other => Err(DaftError::TypeError(format!(
                "cannot infer point dimension from {other:?}"
            ))),
        }
    }
}

impl std::ops::Deref for GeoPointArray {
    type Target = PointArray;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

pub fn import_arrow(data: ArrowData) -> DaftResult<(Field, ArrayRef)> {
    let field = Field::try_from(&data.schema).map_err(|e| DaftError::TypeError(e.to_string()))?;
    let ffi_schema: FFI_ArrowSchema = data.schema.into();
    let ffi_array: FFI_ArrowArray = data.array.into();
    let array_data = unsafe { from_ffi(ffi_array, &ffi_schema) }
        .map_err(|e| DaftError::RuntimeError(e.to_string()))?;
    Ok((field, make_array(array_data)))
}

pub fn export_arrow(field: &Field, array: ArrayRef) -> DaftResult<ArrowData> {
    let schema = ArrowSchema::try_from(field).map_err(|e| DaftError::TypeError(e.to_string()))?;
    let (ffi_array, _ffi_schema) =
        to_ffi(&array.to_data()).map_err(|e| DaftError::RuntimeError(e.to_string()))?;
    Ok(ArrowData {
        schema,
        array: ffi_array.into(),
    })
}
