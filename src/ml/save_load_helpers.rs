// Helper module for conditional serialization
mod save_load_helpers {
    use super::Result;
    use std::path::PathBuf;
    
    pub fn not_available_error() -> Result<()> {
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "Serialization feature not enabled. Enable with --features serialization",
        ))
    }
}
