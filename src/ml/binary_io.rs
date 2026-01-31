// Binary serialization utilities for model parameters
// Replaces serde/serde_json/anyhow dependencies

use std::io::{self, Read, Write};

/// Magic bytes for file format identification: "MNML"
const MAGIC: &[u8; 4] = b"MNML";
/// Format version
const VERSION: u8 = 1;

/// Write a Tensor's raw data to binary format
pub fn write_tensor_data<W: Write>(writer: &mut W, data: &[f32], shape: &[usize]) -> io::Result<()> {
    // Write shape length
    writer.write_all(&(shape.len() as u32).to_le_bytes())?;
    
    // Write shape
    for &dim in shape {
        writer.write_all(&(dim as u32).to_le_bytes())?;
    }
    
    // Write data length
    writer.write_all(&(data.len() as u32).to_le_bytes())?;
    
    // Write data
    for &val in data {
        writer.write_all(&val.to_le_bytes())?;
    }
    
    Ok(())
}

/// Read Tensor data from binary format
pub fn read_tensor_data<R: Read>(reader: &mut R) -> io::Result<(Vec<f32>, Vec<usize>)> {
    // Read shape length
    let mut buf = [0u8; 4];
    reader.read_exact(&mut buf)?;
    let shape_len = u32::from_le_bytes(buf) as usize;
    
    // Read shape
    let mut shape = Vec::with_capacity(shape_len);
    for _ in 0..shape_len {
        reader.read_exact(&mut buf)?;
        shape.push(u32::from_le_bytes(buf) as usize);
    }
    
    // Read data length
    reader.read_exact(&mut buf)?;
    let data_len = u32::from_le_bytes(buf) as usize;
    
    // Read data
    let mut data = Vec::with_capacity(data_len);
    for _ in 0..data_len {
        reader.read_exact(&mut buf)?;
        data.push(f32::from_le_bytes(buf));
    }
    
    Ok((data, shape))
}

/// Write header (magic + version + layer type)
pub fn write_header<W: Write>(writer: &mut W, layer_type: u8) -> io::Result<()> {
    writer.write_all(MAGIC)?;
    writer.write_all(&[VERSION])?;
    writer.write_all(&[layer_type])?;
    Ok(())
}

/// Read and verify header
pub fn read_header<R: Read>(reader: &mut R, expected_type: u8) -> io::Result<()> {
    let mut magic_buf = [0u8; 4];
    reader.read_exact(&mut magic_buf)?;
    
    if &magic_buf != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Invalid file format: magic bytes mismatch",
        ));
    }
    
    let mut version = [0u8; 1];
    reader.read_exact(&mut version)?;
    
    if version[0] != VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Unsupported version: {}", version[0]),
        ));
    }
    
    let mut layer_type = [0u8; 1];
    reader.read_exact(&mut layer_type)?;
    
    if layer_type[0] != expected_type {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("Layer type mismatch: expected {}, got {}", expected_type, layer_type[0]),
        ));
    }
    
    Ok(())
}

// Layer type constants
pub const TYPE_LINEAR: u8 = 1;
pub const TYPE_MM: u8 = 2;
pub const TYPE_BIAS: u8 = 3;
pub const TYPE_QUANTIZED_LINEAR: u8 = 4;
