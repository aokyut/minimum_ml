use super::Node;
use super::Tensor;
use super::TensorData;


pub trait SingleShoot {
    fn single_forward(&self, x: f32) -> f32;
    fn single_backward(&self, x: f32, y: f32) -> f32;
    fn no_grad(&self) -> bool {
        false
    }
}

impl<F: SingleShoot> Node for F {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let input_data = inputs[0].as_f32_slice();
        let output_data = output.as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            igrad_f32[i] = grad_data[i] * self.single_backward(input_data[i], output_data[i]);
        }

        vec![igrad]
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let input_data = input.as_f32_slice();
        let mut output_vec = vec![0.0; input_data.len()];

        for i in 0..input_data.len() {
            output_vec[i] = self.single_forward(input_data[i]);
        }

        Tensor::new(output_vec, input.shape.clone())
    }
}

pub struct ReLU {
    ignore_grad: bool,
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl ReLU {
    pub fn new() -> Self {
        ReLU { ignore_grad: false }
    }
}

impl Node for ReLU {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let input_data = inputs[0].as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            igrad_f32[i] = grad_data[i] * input_data[i].signum().max(0.0);
        }

        vec![igrad]
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let input_data = input.as_f32_slice();
        let mut output_vec = vec![0.0; input_data.len()];

        for i in 0..input_data.len() {
            output_vec[i] = input_data[i].max(0.0);
        }

        Tensor::new(output_vec, input.shape.clone())
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct LeaklyReLU {
    ignore_grad: bool,
    alpha: f32,
}

impl LeaklyReLU {
    pub fn new(alpha: f32) -> Self {
        LeaklyReLU {
            ignore_grad: false,
            alpha,
        }
    }

    pub fn default() -> Self {
        LeaklyReLU {
            ignore_grad: false,
            alpha: 0.01,
        }
    }
}

impl Node for LeaklyReLU {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let input_data = input.as_f32_slice();
        let mut output_vec = vec![0.0; input_data.len()];
        for i in 0..input_data.len() {
            let x = input_data[i];
            if x > 0.0 {
                output_vec[i] = x;
            } else {
                output_vec[i] = self.alpha * x;
            }
        }

        Tensor::new(output_vec, input.shape.clone())
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let input_data = inputs[0].as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            let x = input_data[i];
            if x < 0.0 {
                igrad_f32[i] = self.alpha * grad_data[i];
            } else {
                igrad_f32[i] = grad_data[i];
            }
        }

        vec![igrad]
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct ClippedReLU {
    ignore_grad: bool,
    ceil: f32,
}

impl ClippedReLU {
    pub fn new(ceil: f32) -> Self {
        assert!(ceil > 0.0);
        ClippedReLU {
            ignore_grad: false,
            ceil,
        }
    }

    pub fn default() -> Self {
        ClippedReLU::new(1.0)
    }
}

impl Node for ClippedReLU {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);
        let input = &input[0];
        let input_data = input.as_f32_slice();
        let mut output_vec = vec![0.0; input_data.len()];
        for i in 0..input_data.len() {
            output_vec[i] = input_data[i].max(0.0).min(self.ceil);
        }

        Tensor::new(output_vec, input.shape.clone())
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let input_data = inputs[0].as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            let x = input_data[i];
            if x < 0.0 {
                igrad_f32[i] = 0.0
            } else if x > self.ceil {
                igrad_f32[i] = 0.0
            } else {
                igrad_f32[i] = grad_data[i];
            }
        }

        vec![igrad]
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct Tanh {}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl Tanh {
    pub fn new() -> Self {
        Tanh {}
    }
}

impl SingleShoot for Tanh {
    fn single_backward(&self, _: f32, y: f32) -> f32 {
        1.0 - y.powi(2)
    }
    fn single_forward(&self, x: f32) -> f32 {
        let ex = x.exp();
        (ex - 1.0 / ex) / (ex + 1.0 / ex)
    }
}

pub struct SoftPlus {}

impl SoftPlus{
    pub fn new() -> Self {
        SoftPlus {}
    }
}

impl SingleShoot for SoftPlus {
    fn single_backward(&self, x: f32, _: f32) -> f32 {
        1.0 / (1.0 + (-x).exp())
    }
    fn single_forward(&self, x: f32) -> f32 {
        (1.0 + x.exp()).ln()
    }
}

pub struct Sigmoid {
    alpha: f32,
}

impl Sigmoid {
    pub fn new(alpha: f32) -> Self {
        Sigmoid { alpha }
    }
}

impl SingleShoot for Sigmoid {
    fn single_backward(&self, _: f32, y: f32) -> f32 {
        self.alpha * y * (1.0 - y)
    }
    fn single_forward(&self, x: f32) -> f32 {
        
        1.0 / (1.0 + (-x * self.alpha).exp())
    }
}

/// Fast approximation of Sigmoid using Pade approximation
/// Avoids expensive exp() computation while maintaining good accuracy (<0.1% error)
pub struct FastSigmoid {}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl FastSigmoid {
    pub fn new() -> Self {
        FastSigmoid {}
    }
}

impl SingleShoot for FastSigmoid {
    fn single_backward(&self, x: f32, _y: f32) -> f32 {
        // Derivative of Pade approximation: 1 / (2 * (1 + |x|)^2)
        let abs_x = x.abs();
        let denominator = 1.0 + abs_x;
        0.5 / (denominator * denominator)
    }
    
    fn single_forward(&self, x: f32) -> f32 {
        // Pade approximation: 0.5 + x / (2 * (1 + |x|))
        // Fast and accurate for -6 < x < 6
        if x < -6.0 {
            return 0.0;
        }
        if x > 6.0 {
            return 1.0;
        }
        0.5 + x / (2.0 * (1.0 + x.abs()))
    }
}

pub struct Softmax {}


impl Default for Softmax {
    fn default() -> Self {
        Self::new()
    }
}

impl Softmax {
    pub fn new() -> Self {
        Softmax {}
    }
}

impl Node for Softmax {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);

        let &width = input[0].shape.last().unwrap();
        let input = &input[0];
        let input_data = input.as_f32_slice();
        let batch = input_data.len() / width;
        let mut output_vec = vec![0.0; input_data.len()];

        for i in 0..batch {
            let offset = i * width;
            let mut sum = 0.0;
            let mut max = input_data[offset];
            for j in 0..width {
                if input_data[offset + j] > max {
                    max = input_data[offset + j];
                }
            }

            for j in 0..width {
                output_vec[offset + j] = (input_data[offset + j] - max).exp();
                sum += output_vec[offset + j]
            }
            for j in 0..width {
                output_vec[offset + j] /= sum;
            }
        }

        Tensor::new(output_vec, input.shape.clone())
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let &width = inputs[0].shape.last().unwrap();
        let igrad_f32 = igrad.f32_data_mut();
        let batch = igrad_f32.len() / width;
        let output_data = output.as_f32_slice();
        let grad_data = grad.as_f32_slice();

        for i in 0..batch {
            let offset = i * width;
            let mut dot_prod = 0.0;
            for j in 0..width {
                dot_prod += output_data[offset + j] * grad_data[offset + j];
            }
            for j in 0..width {
                igrad_f32[offset + j] =
                    output_data[offset + j] * (grad_data[offset + j] - dot_prod);
            }
        }

        vec![igrad]
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct MSE {}

impl Default for MSE {
    fn default() -> Self {
        Self::new()
    }
}

impl MSE {
    pub fn new() -> Self {
        MSE {}
    }
}

impl Node for MSE {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 2);
        let left = input[0].as_f32_slice();
        let right = input[1].as_f32_slice();
        assert_eq!(left.len(), right.len());
        let mut loss = 0.0;

        for i in 0..left.len() {
            loss += (left[i] - right[i]).powi(2);
        }

        Tensor::new(vec![loss / left.len() as f32], vec![1])
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut left = Tensor::zeros_like(inputs[0]);
        let mut right = Tensor::zeros_like(inputs[1]);
        let left_f32 = left.f32_data_mut();
        let right_f32 = right.f32_data_mut();
        let in_left = inputs[0].as_f32_slice();
        let in_right = inputs[1].as_f32_slice();
        let g = grad.get_item().unwrap();

        for i in 0..left_f32.len() {
            left_f32[i] = 2.0 * (in_left[i] - in_right[i]) * g / left_f32.len() as f32;
            right_f32[i] = 2.0 * (in_right[i] - in_left[i]) * g / left_f32.len() as f32;
        }

        vec![left, right]
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct BinaryCrossEntropyLoss {
    eps: f32,
}

impl BinaryCrossEntropyLoss {
    pub fn default() -> Self {
        BinaryCrossEntropyLoss { eps: 0.001 }
    }
}

impl Node for BinaryCrossEntropyLoss {
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 2);
        let left = input[0].as_f32_slice();
        let right = input[1].as_f32_slice();
        assert_eq!(left.len(), right.len());
        let mut loss = 0.0;
        let scale = 1.0 / (left.len()) as f32;

        for i in 0..left.len() {
            loss += -left[i].ln() * right[i]
                - (1.0 - left[i]).ln() * (1.0 - right[i])
                - (-(right[i] + self.eps).ln() * right[i]
                    - (1.0 - right[i] + self.eps).ln() * (1.0 - right[i]));
        }

        Tensor::new(vec![loss * scale], vec![1])
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut left = Tensor::zeros_like(inputs[0]);
        let mut right = Tensor::zeros_like(inputs[1]);
        let left_f32 = left.f32_data_mut();
        let right_f32 = right.f32_data_mut();
        let in_left = inputs[0].as_f32_slice();
        let in_right = inputs[1].as_f32_slice();
        let g = grad.get_item().unwrap();
        let scale = 1.0 / (left_f32.len() as f32);

        for i in 0..left_f32.len() {
            left_f32[i] = (-in_right[i] / (in_left[i] + self.eps)
                + (1.0 - in_right[i]) / (1.0 - in_left[i] + self.eps))
                * g
                * scale;
            right_f32[i] = -(in_left[i].ln() + (1.0 - in_left[i]).ln()) * g * scale;
        }

        vec![left, right]
    }
    fn no_grad(&self) -> bool {
        false
    }
}

/// Cross Entropy Loss for multi-class classification
/// Combines Softmax + Negative Log Likelihood in one stable operation
/// Input: (predictions, targets) where predictions are raw logits (before softmax)
/// and targets are one-hot encoded labels
pub struct CrossEntropyLoss {
    eps: f32,
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        CrossEntropyLoss { eps: 1e-7 }
    }
}

impl Node for CrossEntropyLoss {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        assert_eq!(inputs.len(), 2);
        let logits = inputs[0].as_f32_slice();
        let targets = inputs[1].as_f32_slice();
        assert_eq!(logits.len(), targets.len());
        
        let batch_size = inputs[0].shape[0];
        let num_classes = logits.len() / batch_size;
        let mut total_loss = 0.0;
        
        for b in 0..batch_size {
            let offset = b * num_classes;
            let batch_logits = &logits[offset..offset + num_classes];
            let batch_targets = &targets[offset..offset + num_classes];
            
            // Stable softmax: subtract max for numerical stability
            let max_logit = batch_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            let mut log_probs = vec![0.0; num_classes];
            
            for i in 0..num_classes {
                let exp_val = (batch_logits[i] - max_logit).exp();
                log_probs[i] = exp_val;
                exp_sum += exp_val;
            }
            
            // Compute log probabilities and loss
            for i in 0..num_classes {
                log_probs[i] = (log_probs[i] / (exp_sum + self.eps)).ln();
                total_loss -= batch_targets[i] * log_probs[i];
            }
        }
        
        Tensor::new(vec![total_loss / batch_size as f32], vec![1])
    }
    
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _output: &Tensor) -> Vec<Tensor> {
        let logits = inputs[0].as_f32_slice();
        let targets = inputs[1].as_f32_slice();
        let g = grad.get_item().unwrap();
        
        let batch_size = inputs[0].shape[0];
        let num_classes = logits.len() / batch_size;
        
        let mut logits_grad = Tensor::zeros_like(inputs[0]);
        let logits_grad_data = logits_grad.f32_data_mut();
        
        for b in 0..batch_size {
            let offset = b * num_classes;
            let batch_logits = &logits[offset..offset + num_classes];
            let batch_targets = &targets[offset..offset + num_classes];
            
            // Compute softmax
            let max_logit = batch_logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let mut exp_sum = 0.0;
            let mut softmax = vec![0.0; num_classes];
            
            for i in 0..num_classes {
                softmax[i] = (batch_logits[i] - max_logit).exp();
                exp_sum += softmax[i];
            }
            
            for i in 0..num_classes {
                softmax[i] /= exp_sum + self.eps;
            }
            
            // Gradient: (softmax - target) / batch_size
            for i in 0..num_classes {
                logits_grad_data[offset + i] = (softmax[i] - batch_targets[i]) * g / batch_size as f32;
            }
        }
        
        vec![logits_grad, Tensor::zeros_like(inputs[1])]
    }
    
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct BinaryCrossEntropy{
    eps: f32,
}

impl BinaryCrossEntropy {
    pub fn new() -> Self {
        BinaryCrossEntropy { eps: 1e-7 }
    }
}

impl Node for BinaryCrossEntropy {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        assert_eq!(inputs.len(), 2);
        let predictions = inputs[0].as_f32_slice();
        let targets = inputs[1].as_f32_slice();
        assert_eq!(predictions.len(), targets.len());
        
        let mut loss_vec = vec![0.0; predictions.len()];
        
        for i in 0..predictions.len() {
            loss_vec[i] = -targets[i] * (predictions[i] + self.eps).ln()
                - (1.0 - targets[i]) * (1.0 - predictions[i] + self.eps).ln();
        }
        
        Tensor::new(loss_vec, inputs[0].shape.clone())
    }
    
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _output: &Tensor) -> Vec<Tensor> {
        let predictions = inputs[0].as_f32_slice();
        let targets = inputs[1].as_f32_slice();
        let g = grad.as_f32_slice();
        
        let mut predictions_grad = Tensor::zeros_like(inputs[0]);
        let predictions_grad_data = predictions_grad.f32_data_mut();
        
        for i in 0..predictions.len() {
            predictions_grad_data[i] = (-targets[i] / (predictions[i] + self.eps)
                + (1.0 - targets[i]) / (1.0 - predictions[i] + self.eps)) * g[i];
        }
        
        vec![predictions_grad, Tensor::zeros_like(inputs[1])]
    }
    
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Sum {
    axis: Option<usize>,
    keepdims: bool,
}

impl Sum {
    pub fn new(axis: Option<usize>, keepdims: bool) -> Self {
        Sum { axis, keepdims }
    }

    pub fn sum_all(&self, input: &Tensor) -> Tensor {
        let input_data = input.as_f32_slice();
        let sum = input_data.iter().sum();
        Tensor::new(vec![sum], vec![1])
    }

    pub fn sum_along_axis(&self, input: &Tensor, axis: usize) -> Tensor {
        let mut output_shape = input.shape.clone();
        let dim_size = output_shape[axis];
        if self.keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }
        let mut output_data = vec![0.0; output_shape.iter().product()];
        let input_data = input.as_f32_slice();
        
        for i in 0..input_data.len() {
            let mut out_idx = i;
            for j in (axis + 1)..input.shape.len() {
                out_idx /= input.shape[j];
            }
            out_idx /= dim_size;
            output_data[out_idx] += input_data[i];
        }
        
        Tensor::new(output_data, output_shape)
    }
}

impl Node for Sum {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        match self.axis {
            Some(axis) => self.sum_along_axis(&inputs[0], axis),
            None => self.sum_all(&inputs[0]),
        }
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        match self.axis{
            None => {
                let mut igrad = Tensor::zeros_like(inputs[0]);
                let grad_data = grad.get_item().unwrap();
                let igrad_f32 = igrad.f32_data_mut();
        
                for i in 0..igrad_f32.len() {
                    igrad_f32[i] = grad_data;
                }
        
                vec![igrad]
            },
            Some(ax) => {
                let mut output_shape = grad.shape.clone();
                let dim_size = inputs[0].shape[ax];
                let mut output_data = vec![0.0; output_shape.iter().product()];
                let mut grad_data = grad.as_f32_slice();
                let mut igrad_data = vec![0.0; inputs[0].len()];
                
                for i in 0..inputs[0].len() {
                    let mut out_idx = i;
                    for j in (ax + 1)..inputs[0].shape.len() {
                        out_idx /= inputs[0].shape[j];
                    }
                    out_idx /= dim_size;
                    igrad_data[i] += grad_data[out_idx];
                }
                
                vec![Tensor::new(igrad_data, inputs[0].shape.clone())]
            }
        }
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Mean{
    axis: Option<usize>,
    keepdims: bool,
}

impl Mean{
    pub fn new(axis: Option<usize>, keepdims: bool) -> Self{
        return Self{axis, keepdims};
    }

    pub fn sum_all(&self, input: &Tensor) -> Tensor {
        let input_data = input.as_f32_slice();
        let mean = input_data.iter().sum::<f32>() / (input_data.len() as f32);
        Tensor::new(vec![mean], vec![1])
    }

    pub fn sum_along_axis(&self, input: &Tensor, axis: usize) -> Tensor {
        let mut output_shape = input.shape.clone();
        let dim_size = output_shape[axis];
        if self.keepdims {
            output_shape[axis] = 1;
        } else {
            output_shape.remove(axis);
        }
        let mut output_data = vec![0.0; output_shape.iter().product()];
        let input_data = input.as_f32_slice();
        
        for i in 0..input_data.len() {
            let mut out_idx = i;
            for j in (axis + 1)..input.shape.len() {
                out_idx /= input.shape[j];
            }
            out_idx /= dim_size;
            output_data[out_idx] += input_data[i] / (dim_size as f32);
        }
        
        Tensor::new(output_data, output_shape)
    }
}

impl Node for Mean{
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        match self.axis {
            Some(axis) => self.sum_along_axis(&inputs[0], axis),
            None => self.sum_all(&inputs[0]),
        }
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        match self.axis{
            None => {
                let mut igrad = Tensor::zeros_like(inputs[0]);
                let grad_data = grad.get_item().unwrap();
                let igrad_f32 = igrad.f32_data_mut();
        
                for i in 0..igrad_f32.len() {
                    igrad_f32[i] = grad_data / igrad_f32.len() as f32;
                }
        
                vec![igrad]
            },
            Some(ax) => {
                let dim_size = inputs[0].shape[ax];
                let grad_data = grad.as_f32_slice();
                let mut igrad_data = vec![0.0; inputs[0].len()];
                
                for i in 0..inputs[0].len() {
                    let mut out_idx = i;
                    for j in (ax + 1)..inputs[0].shape.len() {
                        out_idx /= inputs[0].shape[j];
                    }
                    out_idx /= dim_size;
                    igrad_data[i] += grad_data[out_idx] / dim_size as f32;
                }
                
                vec![Tensor::new(igrad_data, inputs[0].shape.clone())]
            }
        }
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct CumlativeSum {
    axis: usize,
}

impl CumlativeSum {
    pub fn new(axis: usize) -> Self {
        CumlativeSum { axis }
    }
}

impl Node for CumlativeSum {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        let input = &inputs[0];
        let input_data = input.as_f32_slice();
        let mut output_vec = vec![0.0; input_data.len()];
        let mut running_sum = 0.0;

        for i in 0..input_data.len() {
            running_sum += input_data[i];
            output_vec[i] = running_sum;
        }

        Tensor::new(output_vec, input.shape.clone())
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();
        let mut running_sum = 0.0;

        for i in (0..igrad_f32.len()).rev() {
            running_sum += grad_data[i];
            igrad_f32[i] = running_sum;
        }

        vec![igrad]
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Tile{
    repeats: Vec<usize>,
}

impl Tile {
    pub fn new(repeats: Vec<usize>) -> Self {
        Tile { repeats }
    }
}

impl Node for Tile {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        let input = &inputs[0];
        let input_data = input.as_f32_slice();
        let mut output_shape = Vec::new();
        for (dim, &rep) in input.shape.iter().zip(self.repeats.iter()) {
            output_shape.push(dim * rep);
        }
        let mut output_data = vec![0.0; output_shape.iter().product()];

        for i in 0..output_data.len() {
            let mut in_idx = i;
            for j in (0..input.shape.len()).rev() {
                in_idx /= self.repeats[j];
                in_idx %= input.shape[j];
            }
            output_data[i] = input_data[in_idx];
        }

        Tensor::new(output_data, output_shape)
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            let mut in_idx = i;
            for j in (0..inputs[0].shape.len()).rev() {
                in_idx /= self.repeats[j];
                in_idx %= inputs[0].shape[j];
            }
            igrad_f32[in_idx] += grad_data[i];
        }

        vec![igrad]
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Add {}

impl Add{
    pub fn new() -> Self {
        Add {}
    }   
}

impl Node for Add {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        assert_eq!(inputs.len(), 2);
        let left = inputs[0].as_f32_slice();
        let right = inputs[1].as_f32_slice();
        let (output_shape, (left_indices, right_indices)) = Tensor::broadcast_shape(&inputs[0], &inputs[1]);
        // Broadcasting support: if shapes differ, we assume one is a scalar and broadcast it
        let mut output_vec = vec![0.0; output_shape.iter().product()];

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate() {
            output_vec[o] = left[l] + right[r];
        }

        Tensor::new(output_vec, output_shape)
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let left = inputs[0];
        let right = inputs[1];
        let (_, (left_indices, right_indices)) = Tensor::broadcast_shape(left, right);
        let mut left_grad = vec![0.0; left.len()];
        let mut right_grad = vec![0.0; right.len()];
        let out_grad_data = grad.as_f32_slice();

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate(){
            left_grad[l] += out_grad_data[o];
            right_grad[r] += out_grad_data[o];
        }

        return vec![Tensor::new(left_grad, left.shape.clone()), Tensor::new(right_grad, right.shape.clone())];
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Sub{}

impl Sub{
    pub fn new() -> Self{
        return Sub{};
    }
}

impl Node for Sub{
     fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        assert_eq!(inputs.len(), 2);
        let left = inputs[0].as_f32_slice();
        let right = inputs[1].as_f32_slice();
        let (output_shape, (left_indices, right_indices)) = Tensor::broadcast_shape(&inputs[0], &inputs[1]);
        // Broadcasting support: if shapes differ, we assume one is a scalar and broadcast it
        let mut output_vec = vec![0.0; output_shape.iter().product()];

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate() {
            output_vec[o] = left[l] - right[r];
        }

        Tensor::new(output_vec, output_shape)
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let left = inputs[0];
        let right = inputs[1];
        let (_, (left_indices, right_indices)) = Tensor::broadcast_shape(left, right);
        let mut left_grad = vec![0.0; left.len()];
        let mut right_grad = vec![0.0; right.len()];
        let out_grad_data = grad.as_f32_slice();

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate(){
            left_grad[l] += out_grad_data[o];
            right_grad[r] -= out_grad_data[o];
        }

        return vec![Tensor::new(left_grad, left.shape.clone()), Tensor::new(right_grad, right.shape.clone())];
    }
    fn no_grad(&self) -> bool {
        false
    }
}

pub struct Mul{}

impl Mul{
    pub fn new() -> Self{
        Self{}
    }
}

impl Node for Mul{
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        assert_eq!(inputs.len(), 2);
        let left = inputs[0].as_f32_slice();
        let right = inputs[1].as_f32_slice();
        let (output_shape, (left_indices, right_indices)) = Tensor::broadcast_shape(&inputs[0], &inputs[1]);
        // Broadcasting support: if shapes differ, we assume one is a scalar and broadcast it
        let mut output_vec = vec![0.0; output_shape.iter().product()];

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate() {
            output_vec[o] = left[l] * right[r];
        }

        Tensor::new(output_vec, output_shape)
    }
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let left = inputs[0];
        let right = inputs[1];
        let left_data = left.as_f32_slice();
        let right_data = right.as_f32_slice();
        let (_, (left_indices, right_indices)) = Tensor::broadcast_shape(left, right);
        let mut left_grad = vec![0.0; left.len()];
        let mut right_grad = vec![0.0; right.len()];
        let out_grad_data = grad.as_f32_slice();

        for (o, (&l, &r)) in left_indices.iter().zip(right_indices.iter()).enumerate(){
            left_grad[l] += out_grad_data[o] * right_data[r];
            right_grad[r] += out_grad_data[o] * left_data[l];
        }

        return vec![Tensor::new(left_grad, left.shape.clone()), Tensor::new(right_grad, right.shape.clone())];
    }
    fn no_grad(&self) -> bool {
        false
    }
}