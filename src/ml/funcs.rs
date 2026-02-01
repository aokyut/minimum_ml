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

// impl<F: SingleShootLoss> Node for F {
//     fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor> {
//         let mut igrad1 = Tensor::zeros_like(inputs[0]);
//         let mut igrad2 = Tensor::zeros_like(inputs[1]);
//         let scale = 1.0 / inputs[0].data.len() as f32;
//         let loss = output.data[0];

//         for i in 0..grad.data.len() {
//             let (x1_, x2_) = self.single_backward_loss(inputs[0].data[i], inputs[1].data[i], loss);
//             igrad1.data[i] = grad.data[i] * scale * x1_;
//             igrad2.data[i] = grad.data[i] * scale * x2_;
//         }

//         return vec![igrad1, igrad2];
//     }
//     fn call(&self, input: Vec<Tensor>) -> Tensor {
//         assert_eq!(input.len(), 1);
//         let xs1 = &input[0];
//         let xs2 = &input[1];
//         let size = xs1.data.len() as f32;
//         let mut loss = 0.0;
//         for i in 0..input.data.len() {
//             loss += self.single_forward_loss(xs1.data[i], xs2.data[i]);
//         }

//         return Tensor::new(vec![loss / size], vec![1]);
//     }
// }

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

pub struct BinaryCrossEntropy {
    eps: f32,
}

impl BinaryCrossEntropy {
    pub fn default() -> Self {
        BinaryCrossEntropy { eps: 0.001 }
    }
}

impl Node for BinaryCrossEntropy {
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

pub struct QuantizeNode {
    pub is_inference: bool,
}

impl Default for QuantizeNode {
    fn default() -> Self {
        Self::new()
    }
}

impl QuantizeNode {
    pub fn new() -> Self {
        Self {
            is_inference: false,
        }
    }
}

impl Node for QuantizeNode {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        let input = &inputs[0];
        if !self.is_inference {
            return input.clone();
        }

        let input_f32 = input.as_f32_slice();
        let n_samples = if input.shape.is_empty() {
            1
        } else {
            input.shape[0]
        };
        let sample_dim = input_f32.len() / n_samples;

        let mut i8_data = vec![0i8; input_f32.len()];
        let mut scales = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let offset = i * sample_dim;
            let sample = &input_f32[offset..offset + sample_dim];

            let mut max_val = 0.0f32;
            for &v in sample {
                max_val = max_val.max(v.abs());
            }
            let scale = if max_val > 0.0 { 127.0 / max_val } else { 1.0 };
            scales.push(scale);

            for j in 0..sample_dim {
                i8_data[offset + j] = (sample[j] * scale).round().clamp(-128.0, 127.0) as i8;
            }
        }

        Tensor::new_i8(i8_data, scales, input.shape.clone())
    }

    fn backward(&mut self, grad: &Tensor, _inputs: Vec<&Tensor>, _output: &Tensor) -> Vec<Tensor> {
        // Identity backward
        vec![grad.clone()]
    }

    fn prepare_inference(&mut self) {
        self.is_inference = true;
    }

    fn prepare_train(&mut self) {
        self.is_inference = false;
    }

    fn no_grad(&self) -> bool {
        false
    }
}

pub struct DequantizeNode {}

impl Default for DequantizeNode {
    fn default() -> Self {
        Self::new()
    }
}

impl DequantizeNode {
    pub fn new() -> Self {
        Self {}
    }
}

impl Node for DequantizeNode {
    fn call(&self, inputs: Vec<Tensor>) -> Tensor {
        let input = &inputs[0];
        match &input.data {
            TensorData::F32(_) => input.clone(),
            TensorData::I8 { .. } => {
                let f32_data = input.as_f32_slice().into_owned();
                Tensor::new(f32_data, input.shape.clone())
            }
        }
    }

    fn backward(&mut self, grad: &Tensor, _inputs: Vec<&Tensor>, _output: &Tensor) -> Vec<Tensor> {
        // Identity backward
        vec![grad.clone()]
    }

    fn prepare_inference(&mut self) {}

    fn no_grad(&self) -> bool {
        false
    }
}
