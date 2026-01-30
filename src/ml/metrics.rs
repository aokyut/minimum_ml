use super::Tensor;

/// Classification metrics for evaluating model performance
pub struct ClassificationMetrics;

impl ClassificationMetrics {
    /// Calculates accuracy for classification tasks.
    ///
    /// # Arguments
    /// * `predictions` - Model output, shape [batch, n_classes] or [batch]
    /// * `targets` - Ground truth, shape [batch, n_classes] (one-hot) or [batch] (labels)
    ///
    /// # Returns
    /// Accuracy as f32 (0.0 to 1.0)
    ///
    /// # Examples
    /// ```
    /// // One-hot targets
    /// let preds = Tensor::new(vec![0.1, 0.9, 0.8, 0.2], vec![2, 2]);
    /// let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);
    /// let acc = ClassificationMetrics::accuracy(&preds, &targets);
    /// ```
    pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
        assert_eq!(
            predictions.shape.len(),
            targets.shape.len(),
            "Predictions and targets must have same number of dimensions"
        );

        let pred_labels = if predictions.shape.len() == 1 {
            // Already label indices
            predictions.clone()
        } else {
            // Need to apply argmax on last dimension
            let last_dim = predictions.shape.len() - 1;
            predictions.argmax(Some(last_dim))
        };

        let true_labels = if targets.shape.len() == 1 {
            // Already label indices
            targets.clone()
        } else {
            // One-hot encoded, need argmax
            let last_dim = targets.shape.len() - 1;
            targets.argmax(Some(last_dim))
        };

        let pred_data = pred_labels.as_f32_slice();
        let true_data = true_labels.as_f32_slice();

        assert_eq!(
            pred_data.len(),
            true_data.len(),
            "Predictions and targets must have same batch size"
        );

        let mut correct = 0;
        for (a, b) in pred_data.iter().zip(true_data.iter()) {
            if (a - b).abs() < 1e-6 {
                correct += 1;
            }
        }

        correct as f32 / pred_data.len() as f32
    }

    /// Calculates top-k accuracy for classification tasks.
    ///
    /// # Arguments
    /// * `predictions` - Model output, shape [batch, n_classes]
    /// * `targets` - Ground truth labels, shape [batch] or [batch, n_classes]
    /// * `k` - Number of top predictions to consider
    ///
    /// # Returns
    /// Top-k accuracy as f32 (0.0 to 1.0)
    pub fn top_k_accuracy(predictions: &Tensor, targets: &Tensor, k: usize) -> f32 {
        assert!(
            predictions.shape.len() == 2,
            "Predictions must be 2D [batch, n_classes]"
        );
        let batch_size = predictions.shape[0];
        let n_classes = predictions.shape[1];

        assert!(k <= n_classes, "k must be <= number of classes");

        let true_labels = if targets.shape.len() == 1 {
            targets.clone()
        } else {
            targets.argmax(Some(1))
        };

        let pred_data = predictions.as_f32_slice();
        let true_data = true_labels.as_f32_slice();

        let mut correct = 0;
        for i in 0..batch_size {
            let offset = i * n_classes;
            let row = &pred_data[offset..offset + n_classes];

            // Get top-k indices
            let mut indexed: Vec<(usize, f32)> = row.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_indices: Vec<usize> = indexed.iter().take(k).map(|(idx, _)| *idx).collect();

            let true_label = true_data[i] as usize;
            if top_k_indices.contains(&true_label) {
                correct += 1;
            }
        }

        correct as f32 / batch_size as f32
    }
}

/// Convenience function for accuracy calculation
pub fn accuracy(predictions: &Tensor, targets: &Tensor) -> f32 {
    ClassificationMetrics::accuracy(predictions, targets)
}

/// Convenience function for top-k accuracy calculation
pub fn top_k_accuracy(predictions: &Tensor, targets: &Tensor, k: usize) -> f32 {
    ClassificationMetrics::top_k_accuracy(predictions, targets, k)
}
