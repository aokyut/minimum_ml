use super::xiver_vec;
use super::{Node, Result, Tensor, TensorData};


pub struct Parameter{
    pub tensor: Tensor,
    pub grad: Option<Tensor>,
    pub ignore_grad: bool,
}

impl Parameter {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor,
            grad: None,
            ignore_grad: false,
        }
    }

    pub fn set_ignore(&mut self) {
        self.ignore_grad = true;
    }
}

impl Node for Parameter {
    fn backward(&mut self, grad: &Tensor, _: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        if let Some(_grad) = self.grad.as_mut() {
            *_grad += grad;
        } else {
            self.grad = Some(grad.clone());
        }
        vec![]
    }

    fn call(&self, _: Vec<Tensor>) -> Tensor {
        self.tensor.clone()
    }

    fn prepare_inference(&mut self) {}
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn has_params(&self) -> bool {
        !self.ignore_grad
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        Some(vec![self.grad.as_ref().unwrap()])
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        let tensor_f32 = self.tensor.f32_data_mut();
        let up_f32 = update[0].as_f32_slice();
        for i in 0..tensor_f32.len() {
            tensor_f32[i] += up_f32[i];
        }

        self.grad = None;
    }

    fn print(&self) {
        println!("param:{:?}", self.tensor)
    }

    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        write_header(&mut writer, TYPE_PARAMETER)?;
        write_tensor_data(&mut writer, self.tensor.as_f32_slice().as_ref(), &self.tensor.shape)?;

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        read_header(&mut reader, TYPE_PARAMETER)?;
        
        let (data, shape) = read_tensor_data(&mut reader)?;
        self.tensor = Tensor {
            data: TensorData::F32(data),
            shape,
        };

        Ok(())
    }
}

pub struct Linear {
    pub w: Tensor,
    pub b: Tensor,
    pub height: usize,
    pub width: usize,
    pub w_grad: Option<Tensor>,
    pub b_grad: Option<Tensor>,
    pub ignore_grad: bool,
}

impl Linear {
    pub fn new(w: Tensor, b: Tensor) -> Self {
        assert_eq!(w.shape.len(), 2);
        assert_eq!(b.shape.len(), 1);
        assert_eq!(b.shape[0], w.shape[0]);

        let height = w.shape[0];
        let width = w.shape[1];
        Self {
            w,
            b,
            height,
            width,
            w_grad: None,
            b_grad: None,
            ignore_grad: false,
        }
    }

    pub fn auto(input_size: usize, output_size: usize) -> Self {
        let weight = xiver_vec(input_size, output_size * input_size);
        let weight = Tensor::new(weight, vec![output_size, input_size]);
        let b = Tensor::zeros(vec![output_size]);
        Linear::new(weight, b)
    }

    pub fn set_ignore(&mut self) {
        self.ignore_grad = true;
    }
}

impl Node for Linear {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.height;

        // Calculate batch size by flattening all dimensions except the last
        let input_data = input.as_f32_slice();
        let batch = input_data.len() / in_features;

        let mut w_grad = Tensor::zeros_like(&self.w);
        let mut b_grad = Tensor::zeros_like(&self.b);
        let mut input_grad = Tensor::zeros_like(input);

        let grad_data = grad.as_f32_slice();
        let input_data = input.as_f32_slice();
        let w_data = self.w.as_f32_slice();

        let w_grad_f32 = w_grad.f32_data_mut();
        let b_grad_f32 = b_grad.f32_data_mut();
        let input_grad_f32 = input_grad.f32_data_mut();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_grad = b * out_features;
            for i in 0..out_features {
                let gi = grad_data[offset_grad + i];
                b_grad_f32[i] += gi;
                for j in 0..in_features {
                    w_grad_f32[i * in_features + j] += gi * input_data[offset_input + j];
                    input_grad_f32[offset_input + j] += gi * w_data[i * in_features + j];
                }
            }
        }

        if let Some(_w_grad) = self.w_grad.as_mut() {
            *_w_grad += w_grad;
        } else {
            self.w_grad = Some(w_grad);
        }
        if let Some(_b_grad) = self.b_grad.as_mut() {
            *_b_grad += b_grad;
        } else {
            self.b_grad = Some(b_grad);
        }

        vec![input_grad]
    }

    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];

        let input_f32 = input.as_f32_slice();
        let in_features = *input.shape.last().unwrap();
        assert_eq!(in_features, self.width);

        // Calculate batch size by flattening all dimensions except the last
        let batch = input_f32.len() / in_features;

        let mut ans_shape = input.shape.clone();
        *ans_shape.last_mut().unwrap() = self.height;
        let mut ans_data = vec![0.0; batch * self.height];
        let w_f32 = self.w.as_f32_slice();
        let b_f32 = self.b.as_f32_slice();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_ans = b * self.height;
            for i in 0..self.height {
                let mut sum = b_f32[i];
                for j in 0..in_features {
                    sum += input_f32[offset_input + j] * w_f32[i * in_features + j];
                }
                ans_data[offset_ans + i] = sum;
            }
        }
        Tensor::new(ans_data, ans_shape)
    }

    fn prepare_inference(&mut self) {
        // Linear is now pure F32, no-op or specific F32 optimizations if needed
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn has_params(&self) -> bool {
        !self.ignore_grad
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        Some(vec![
            self.w_grad.as_ref().unwrap(),
            self.b_grad.as_ref().unwrap(),
        ])
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        let w_f32 = self.w.f32_data_mut();
        let up_w = update[0].as_f32_slice();
        for i in 0..w_f32.len() {
            w_f32[i] += up_w[i];
        }

        let b_f32 = self.b.f32_data_mut();
        let up_b = update[1].as_f32_slice();
        for i in 0..b_f32.len() {
            b_f32[i] += up_b[i];
        }

        self.w_grad = None;
        self.b_grad = None;
    }

    fn print(&self) {
        println!("w:{:?}, b:{:?}", self.w, self.b)
    }

    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        write_header(&mut writer, TYPE_LINEAR)?;
        write_tensor_data(&mut writer, self.w.as_f32_slice().as_ref(), &self.w.shape)?;
        write_tensor_data(&mut writer, self.b.as_f32_slice().as_ref(), &self.b.shape)?;

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        read_header(&mut reader, TYPE_LINEAR)?;
        
        let (w_data, w_shape) = read_tensor_data(&mut reader)?;
        let (b_data, b_shape) = read_tensor_data(&mut reader)?;

        self.w = Tensor {
            data: TensorData::F32(w_data),
            shape: w_shape,
        };
        self.b = Tensor {
            data: TensorData::F32(b_data),
            shape: b_shape,
        };

        Ok(())
    }
}

pub struct MM {
    pub w: Tensor,
    pub height: usize,
    pub width: usize,
    pub w_grad: Option<Tensor>,
    pub ignore_grad: bool,
}

impl MM {
    pub fn new(w: Tensor) -> Self {
        assert_eq!(w.shape.len(), 2);

        let height = w.shape[0];
        let width = w.shape[1];
        Self {
            w,
            height,
            width,
            w_grad: None,
            ignore_grad: false,
        }
    }

    pub fn auto(input_size: usize, output_size: usize) -> Self {
        let weight = xiver_vec(output_size, output_size * input_size);
        let weight = Tensor::new(weight, vec![output_size, input_size]);
        // println!("weight:{:#?}, size:{}", weight.shape, weight.data[0]);
        MM::new(weight)
    }

    pub fn set_ignore(&mut self) {
        self.ignore_grad = true;
    }
}

impl Node for MM {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.height;

        // Calculate batch size by flattening all dimensions except the last
        let input_data = input.as_f32_slice();
        let batch = input_data.len() / in_features;

        let mut w_grad = Tensor::zeros_like(&self.w);
        let mut input_grad = Tensor::zeros_like(input);

        let grad_data = grad.as_f32_slice();
        let input_data = input.as_f32_slice();
        let w_data = self.w.as_f32_slice();
        let w_grad_f32 = w_grad.f32_data_mut();
        let input_grad_f32 = input_grad.f32_data_mut();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_grad = b * out_features;
            for i in 0..out_features {
                let gi = grad_data[offset_grad + i];
                for j in 0..in_features {
                    w_grad_f32[i * in_features + j] += gi * input_data[offset_input + j];
                    input_grad_f32[offset_input + j] += gi * w_data[i * in_features + j];
                }
            }
        }

        if let Some(_w_grad) = self.w_grad.as_mut() {
            *_w_grad += w_grad;
        } else {
            self.w_grad = Some(w_grad);
        }

        vec![input_grad]
    }

    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];

        let input_f32 = input.as_f32_slice();
        let in_features = *input.shape.last().unwrap();
        assert_eq!(in_features, self.width);

        // Calculate batch size by flattening all dimensions except the last
        let batch = input_f32.len() / in_features;

        let mut ans_shape = input.shape.clone();
        *ans_shape.last_mut().unwrap() = self.height;
        let mut ans_data = vec![0.0; batch * self.height];
        let w_f32 = self.w.as_f32_slice();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_ans = b * self.height;
            for i in 0..self.height {
                let mut sum = 0.0;
                for j in 0..in_features {
                    sum += input_f32[offset_input + j] * w_f32[i * in_features + j];
                }
                ans_data[offset_ans + i] = sum;
            }
        }
        Tensor::new(ans_data, ans_shape)
    }

    fn prepare_inference(&mut self) {}
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn has_params(&self) -> bool {
        !self.ignore_grad
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        Some(vec![self.w_grad.as_ref().unwrap()])
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        let w_f32 = self.w.f32_data_mut();
        let up_w = update[0].as_f32_slice();
        for i in 0..w_f32.len() {
            w_f32[i] += up_w[i];
        }

        self.w_grad = None;
    }

    fn print(&self) {
        println!("w:{:?}", self.w);
    }

    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        write_header(&mut writer, TYPE_MM)?;
        write_tensor_data(&mut writer, self.w.as_f32_slice().as_ref(), &self.w.shape)?;

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        read_header(&mut reader, TYPE_MM)?;
        
        let (w_data, w_shape) = read_tensor_data(&mut reader)?;
        self.w = Tensor {
            data: TensorData::F32(w_data),
            shape: w_shape,
        };

        Ok(())
    }
}

pub struct Bias {
    pub b: Tensor,
    pub b_grad: Option<Tensor>,
    height: usize,
    ignore_grad: bool,
}

impl Bias {
    pub fn new(b: Tensor) -> Self {
        assert_eq!(b.shape.len(), 1);

        let height = b.shape[0];
        Self {
            b,
            height,
            b_grad: None,
            ignore_grad: false,
        }
    }

    pub fn auto(output_size: usize) -> Self {
        let b = Tensor::zeros(vec![output_size]);
        Bias::new(b)
    }

    pub fn set_ignore(&mut self) {
        self.ignore_grad = true;
    }
}

impl Node for Bias {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let features = *input.shape.last().unwrap();
        assert_eq!(features, self.height);

        // Calculate batch size by flattening all dimensions except the last
        let grad_data = grad.as_f32_slice();
        let batch = grad_data.len() / features;

        let mut b_grad = Tensor::zeros_like(&self.b);
        let input_grad = Tensor::zeros_like(input);

        let grad_data = grad.as_f32_slice();
        let b_grad_f32 = b_grad.f32_data_mut();

        for b_idx in 0..batch {
            let offset_grad = b_idx * self.height;
            for i in 0..self.height {
                b_grad_f32[i] += grad_data[offset_grad + i];
            }
        }

        if let Some(_b_grad) = self.b_grad.as_mut() {
            *_b_grad += b_grad;
        } else {
            self.b_grad = Some(b_grad);
        }

        vec![input_grad]
    }

    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];
        let input_f32 = input.as_f32_slice();
        let features = *input.shape.last().unwrap();
        assert_eq!(features, self.height);

        // Calculate batch size by flattening all dimensions except the last
        let batch = input_f32.len() / features;

        let ans_shape = input.shape.clone();
        let mut ans_data = vec![0.0; batch * self.height];
        let b_f32 = self.b.as_f32_slice();

        for b_idx in 0..batch {
            let offset = b_idx * self.height;
            for i in 0..self.height {
                ans_data[offset + i] = input_f32[offset + i] + b_f32[i];
            }
        }
        Tensor::new(ans_data, ans_shape)
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn has_params(&self) -> bool {
        !self.ignore_grad
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        Some(vec![self.b_grad.as_ref().unwrap()])
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        let b_f32 = self.b.f32_data_mut();
        let up_b = update[0].as_f32_slice();
        for i in 0..b_f32.len() {
            b_f32[i] += up_b[i];
        }

        self.b_grad = None;
    }

    fn print(&self) {
        println!("b:{:?}", self.b)
    }

    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        write_header(&mut writer, TYPE_BIAS)?;
        write_tensor_data(&mut writer, self.b.as_f32_slice().as_ref(), &self.b.shape)?;

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        read_header(&mut reader, TYPE_BIAS)?;
        
        let (b_data, b_shape) = read_tensor_data(&mut reader)?;
        self.b = Tensor {
            data: TensorData::F32(b_data),
            shape: b_shape,
        };

        Ok(())
    }
}

pub struct BatchNorm{
    eps: f32,
    mu: Option<Tensor>,          // batch mean (used in backward / inference)
    delta: Option<Tensor>,       // stddev+eps (denominator)
    pub gamma: Option<Tensor>,       // scale parameter
    pub beta: Option<Tensor>,        // shift parameter
    pub gamma_grad: Option<Tensor>,  // gradient for gamma
    pub beta_grad: Option<Tensor>,   // gradient for beta
    pub momentum: f32,
    is_train: bool,
}

impl BatchNorm{
    pub fn new() -> Self{
        Self {
            eps: 1e-8,
            mu: None,
            delta: None,
            gamma: None,
            beta: None,
            gamma_grad: None,
            beta_grad: None,
            momentum: 0.99,
            is_train: false,
        }
    }

    /// create a batch‑norm layer with learnable parameters
    pub fn auto(num_features: usize) -> Self {
        let mut bn = Self::new();
        bn.gamma = Some(Tensor::ones(vec![num_features]));
        bn.beta = Some(Tensor::zeros(vec![num_features]));
        bn.mu = Some(Tensor::zeros(vec![num_features]));
        bn.delta = Some(Tensor::ones(vec![num_features]));
        bn
    }
}

impl Node for BatchNorm{
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        // Only a single input tensor is expected
        let input = inputs[0];
        let features = *input.shape.last().unwrap();
        let batch = input.as_f32_slice().len() / features;

        let x = input.as_f32_slice();
        let dy = grad.as_f32_slice();

        // recompute mean and delta from input; this mirrors the forward pass during training
        let mut mu = vec![0.0; features];
        for b in 0..batch {
            let base = b * features;
            for i in 0..features {
                mu[i] += x[base + i];
            }
        }
        for i in 0..features {
            mu[i] /= batch as f32;
        }

        for (m, m_) in self.mu.as_mut().unwrap().f32_data_mut().iter_mut().zip(mu.iter()){
            *m = *m * self.momentum + (1.0 - self.momentum) * m_;
        }

        let mut delta = vec![0.0; features];
        for b in 0..batch {
            let base = b * features;
            for i in 0..features {
                let diff = x[base + i] - mu[i];
                delta[i] += diff * diff;
            }
        }
        for i in 0..features {
            delta[i] = (delta[i] / batch as f32 + self.eps).sqrt();
        }

        for (d, d_) in self.delta.as_mut().unwrap().f32_data_mut().iter_mut().zip(delta.iter()){
            *d = *d * self.momentum + d_ * (1.0 - self.momentum);
        }

        // compute gradients for gamma and beta
        let mut dgamma = vec![0.0; features];
        let mut dbeta = vec![0.0; features];

        for b in 0..batch {
            let base = b * features;
            for i in 0..features {
                let idx = base + i;
                dbeta[i] += dy[idx];
                if self.gamma.is_some() {
                    dgamma[i] += dy[idx] * (x[idx] - mu[i]) / delta[i];
                }
            }
        }

        if self.gamma.is_some() {
            if let Some(g_grad) = self.gamma_grad.as_mut() {
                let gdat = g_grad.f32_data_mut();
                for i in 0..features {
                    gdat[i] += dgamma[i];
                }
            } else {
                self.gamma_grad = Some(Tensor::new(dgamma.clone(), vec![features]));
            }
        }
        if self.beta.is_some() {
            if let Some(b_grad) = self.beta_grad.as_mut() {
                let bdat = b_grad.f32_data_mut();
                for i in 0..features {
                    bdat[i] += dbeta[i];
                }
            } else {
                self.beta_grad = Some(Tensor::new(dbeta.clone(), vec![features]));
            }
        }

        // compute input gradient
        let mut dx = Tensor::zeros_like(input);
        let dx_data = dx.f32_data_mut();

        // precompute means over batch
        let mut mean_dy = vec![0.0; features];
        let mut mean_dy_xmu = vec![0.0; features];

        for b in 0..batch {
            let base = b * features;
            for i in 0..features {
                let idx = base + i;
                mean_dy[i] += dy[idx];
                mean_dy_xmu[i] += dy[idx] * (x[idx] - mu[i]);
            }
        }
        for i in 0..features {
            mean_dy[i] /= batch as f32;
            mean_dy_xmu[i] /= batch as f32;
        }

        for b in 0..batch {
            let base = b * features;
            for i in 0..features {
                let idx = base + i;
                let g = if let Some(g) = &self.gamma { g.as_f32_slice()[i] } else { 1.0 };
                dx_data[idx] = g
                    * (1.0 / delta[i])
                    * (dy[idx]
                        - mean_dy[i]
                        - (x[idx] - mu[i]) * (mean_dy_xmu[i] / (delta[i] * delta[i])));
            }
        }

        vec![dx]
    }

    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];
        let data = input.as_f32_slice();
        let features = *input.shape.last().unwrap();
        let batch = data.len() / features;

        let mut out = vec![0.0; data.len()];

        if self.is_train {
            // compute mean over batch
            let mut mu = vec![0.0; features];
            for b in 0..batch {
                let base = b * features;
                for i in 0..features {
                    mu[i] += data[base + i];
                }
            }
            for i in 0..features {
                mu[i] /= batch as f32;
            }

            // compute variance and denominator
            let mut delta = vec![0.0; features];
            for b in 0..batch {
                let base = b * features;
                for i in 0..features {
                    let diff = data[base + i] - mu[i];
                    delta[i] += diff * diff;
                }
            }
            for i in 0..features {
                delta[i] = (delta[i] / batch as f32 + self.eps).sqrt();
            }

            // normalize, scale, shift
            for b in 0..batch {
                let base = b * features;
                for i in 0..features {
                    let idx = base + i;
                    let mut v = (data[idx] - mu[i]) / delta[i];
                    if let Some(g) = &self.gamma {
                        v *= g.as_f32_slice()[i];
                    }
                    if let Some(bias) = &self.beta {
                        v += bias.as_f32_slice()[i];
                    }
                    out[idx] = v;
                }
            }
        } else {
            // inference mode uses stored statistics
            let mu = self.mu.as_ref().expect("BatchNorm missing mean for inference");
            let mu = mu.as_f32_slice();
            let delta = self.delta.as_ref().expect("BatchNorm missing delta for inference");
            let delta = delta.as_f32_slice();


            for b in 0..batch {
                let base = b * features;
                for i in 0..features {
                    let idx = base + i;
                    let mut v = (data[idx] - mu[i]) / delta[i];
                    if let Some(g) = &self.gamma {
                        v *= g.as_f32_slice()[i];
                    }
                    if let Some(bias) = &self.beta {
                        v += bias.as_f32_slice()[i];
                    }
                    out[idx] = v;
                }
            }
        }

        Tensor::new(out, input.shape.clone())
    }

    fn prepare_inference(&mut self) {
        self.is_train = false;
    }

    fn prepare_train(&mut self) {
        self.is_train = true;
    }

    fn no_grad(&self) -> bool {
        // if not training and there are no learnable parameters
        !self.is_train && self.gamma.is_none() && self.beta.is_none()
    }

    fn has_params(&self) -> bool {
        self.gamma.is_some() || self.beta.is_some()
    }

    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        let mut v = Vec::new();
        if let Some(g) = &self.gamma_grad {
            v.push(g);
        }
        if let Some(b) = &self.beta_grad {
            v.push(b);
        }
        if v.is_empty() { None } else { Some(v) }
    }

    fn apply_update(&mut self, update: Vec<Tensor>) {
        let mut idx = 0;
        if self.gamma.is_some() {
            let u = &update[idx];
            let gdat = self.gamma.as_mut().unwrap().f32_data_mut();
            let up = u.as_f32_slice();
            for i in 0..gdat.len() {
                gdat[i] += up[i];
            }
            self.gamma_grad = None;
            idx += 1;
        }
        if self.beta.is_some() {
            let u = &update[idx];
            let bdat = self.beta.as_mut().unwrap().f32_data_mut();
            let up = u.as_f32_slice();
            for i in 0..bdat.len() {
                bdat[i] += up[i];
            }
            self.beta_grad = None;
        }
    }

    fn print(&self) {
        println!(
            "BatchNorm mu:{:?} delta:{:?} gamma:{:?} beta:{:?}",
            self.mu, self.delta, self.gamma, self.beta
        )
    }

    fn save_param(&self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufWriter;

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        write_header(&mut writer, TYPE_BATCHNORM)?;

        // write gamma, beta, mu, delta in order - allow empty vectors
        if let Some(g) = &self.gamma {
            write_tensor_data(&mut writer, g.as_f32_slice().as_ref(), &g.shape)?;
        } else {
            write_tensor_data(&mut writer, &[], &[])?;
        }
        if let Some(b) = &self.beta {
            write_tensor_data(&mut writer, b.as_f32_slice().as_ref(), &b.shape)?;
        } else {
            write_tensor_data(&mut writer, &[], &[])?;
        }
        if let Some(mu) = &self.mu {
            write_tensor_data(&mut writer, mu.as_f32_slice().as_ref(), &mu.shape)?;
        } else {
            write_tensor_data(&mut writer, &[], &[])?;
        }
        if let Some(d) = &self.delta {
            write_tensor_data(&mut writer, d.as_f32_slice().as_ref(), &d.shape)?;
        } else {
            write_tensor_data(&mut writer, &[], &[])?;
        }

        Ok(())
    }

    fn load_param(&mut self, path: std::path::PathBuf) -> Result<()> {
        use super::binary_io::*;
        use std::fs::File;
        use std::io::BufReader;

        let file = File::open(path)?;
        let mut reader = BufReader::new(file);

        read_header(&mut reader, TYPE_BATCHNORM)?;

        let (g_data, g_shape) = read_tensor_data(&mut reader)?;
        self.gamma = if g_shape.is_empty() {
            None
        } else {
            Some(Tensor { data: TensorData::F32(g_data), shape: g_shape })
        };

        let (b_data, b_shape) = read_tensor_data(&mut reader)?;
        self.beta = if b_shape.is_empty() {
            None
        } else {
            Some(Tensor { data: TensorData::F32(b_data), shape: b_shape })
        };

        let (mu_data, mu_shape) = read_tensor_data(&mut reader)?;
        self.mu = if mu_shape.is_empty() {
            None
        } else {
            Some(Tensor { data: TensorData::F32(mu_data), shape: mu_shape })
        };

        let (d_data, d_shape) = read_tensor_data(&mut reader)?;
        self.delta = if d_shape.is_empty() {
            None
        } else {
            Some(Tensor { data: TensorData::F32(d_data), shape: d_shape })
        };

        Ok(())
    }
}
