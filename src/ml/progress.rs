use indicatif::{ProgressBar, ProgressStyle};

/// Training progress bar wrapper
pub struct TrainingProgress {
    epoch_bar: ProgressBar,
    batch_bar: ProgressBar,
}

impl TrainingProgress {
    pub fn new(num_epochs: usize, batches_per_epoch: usize) -> Self {
        let epoch_bar = ProgressBar::new(num_epochs as u64);
        epoch_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] Epoch {pos}/{len} {bar:40.cyan/blue} {msg}")
                .unwrap()
                .progress_chars("##-"),
        );

        let batch_bar = ProgressBar::new(batches_per_epoch as u64);
        batch_bar.set_style(
            ProgressStyle::default_bar()
                .template("  [{elapsed_precise}] Batch {pos}/{len} {bar:40.green/yellow} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        Self {
            epoch_bar,
            batch_bar,
        }
    }

    pub fn start_epoch(&self) {
        self.batch_bar.reset();
    }

    pub fn update_batch(&self, loss: f32) {
        self.batch_bar.set_message(format!("loss: {:.4}", loss));
        self.batch_bar.inc(1);
    }

    pub fn finish_epoch(&self, epoch: usize, test_acc: f32, avg_loss: f32) {
        self.epoch_bar.set_message(format!(
            "acc: {:.2}%, loss: {:.4}",
            test_acc * 100.0,
            avg_loss
        ));
        self.epoch_bar.inc(1);
    }

    pub fn finish(&self) {
        self.batch_bar.finish_and_clear();
        self.epoch_bar.finish_with_message("Training complete!");
    }
}
