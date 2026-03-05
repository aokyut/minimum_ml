//! Simple training progress display without external dependencies

use std::io::{self, Write};

/// Simple training progress tracker
pub struct TrainingProgress {
    num_epochs: usize,
    batches_per_epoch: usize,
    current_epoch: usize,
    current_batch: usize,
}

impl TrainingProgress {
    pub fn new(num_epochs: usize, batches_per_epoch: usize) -> Self {
        Self {
            num_epochs,
            batches_per_epoch,
            current_epoch: 0,
            current_batch: 0,
        }
    }

    pub fn start_epoch(&mut self) {
        self.current_epoch += 1;
        self.current_batch = 0;
    }

    pub fn update_batch(&mut self, loss: f32) {
        self.current_batch += 1;
        print!(
            "\r  Epoch [{}/{}] Batch [{}/{}] loss: {:.4}",
            self.current_epoch,
            self.num_epochs,
            self.current_batch,
            self.batches_per_epoch,
            loss
        );
        io::stdout().flush().ok();
    }

    pub fn finish_epoch(&mut self, _epoch: usize, test_acc: f32, avg_loss: f32) {
        println!(
            "\r  Epoch [{}/{}] Completed - acc: {:.2}%, loss: {:.4}",
            self.current_epoch,
            self.num_epochs,
            test_acc * 100.0,
            avg_loss
        );
    }

    pub fn finish(&self) {
        println!("\nTraining complete!");
    }
}
