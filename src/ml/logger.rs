use chrono::Local;
use std::path::PathBuf;
use tensorboard_rs::summary_writer::SummaryWriter;

/// TensorBoard logger for training metrics
pub struct TensorBoardLogger {
    writer: SummaryWriter,
    step: usize,
}

impl TensorBoardLogger {
    /// Create a new TensorBoard logger
    /// Logs will be saved to runs/{timestamp} by default
    pub fn new() -> Self {
        let timestamp = Local::now().format("%Y%m%d-%H%M%S");
        let log_dir = format!("runs/{}", timestamp);
        Self::with_log_dir(&log_dir)
    }

    /// Create a logger with a custom log directory
    pub fn with_log_dir(log_dir: &str) -> Self {
        let path = PathBuf::from(log_dir);
        let writer = SummaryWriter::new(&path);
        Self { writer, step: 0 }
    }

    /// Log a scalar value
    pub fn log_scalar(&mut self, tag: &str, value: f32) {
        self.writer.add_scalar(tag, value, self.step);
    }

    /// Log multiple scalars at once
    pub fn log_scalars(&mut self, values: &[(&str, f32)]) {
        for (tag, value) in values {
            self.log_scalar(tag, *value);
        }
    }

    /// Increment the global step
    pub fn next_step(&mut self) {
        self.step += 1;
    }

    /// Get current step
    pub fn get_step(&self) -> usize {
        self.step
    }

    /// Flush pending writes
    pub fn flush(&mut self) {
        self.writer.flush();
    }
}

impl Drop for TensorBoardLogger {
    fn drop(&mut self) {
        self.writer.flush();
    }
}
