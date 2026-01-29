use crate::ml::Node;
use crate::ml::{Tensor, TensorData};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize)]
pub struct Quantize {
    is_inference: bool,
}

impl Quantize {
    pub fn new() -> Self {
        return Quantize {
            is_inference: false,
        };
    }
}

impl Node for Quantize {
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

pub struct Dequantize {}

impl Dequantize {
    pub fn new() -> Self {
        return Dequantize {};
    }
}

impl Node for Dequantize {
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

    fn no_grad(&self) -> bool {
        false
    }
}

pub struct QReLU {
    ignore_grad: bool,
    is_inference: bool,
}

impl QReLU {
    pub fn new() -> Self {
        return QReLU {
            ignore_grad: false,
            is_inference: false,
        };
    }
}

impl Node for QReLU {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut igrad = Tensor::zeros_like(inputs[0]);
        let grad_data = grad.as_f32_slice();
        let input_data = inputs[0].as_f32_slice();
        let igrad_f32 = igrad.f32_data_mut();

        for i in 0..grad_data.len() {
            igrad_f32[i] = grad_data[i] * input_data[i].signum().max(0.0);
        }

        return vec![igrad];
    }
    fn call(&self, input: Vec<Tensor>) -> Tensor {
        assert_eq!(input.len(), 1);

        if !self.is_inference {
            let input = &input[0];
            let input_data = input.as_f32_slice();
            let mut output_vec = vec![0.0; input_data.len()];

            for i in 0..input_data.len() {
                output_vec[i] = input_data[i].max(0.0);
            }

            return Tensor::new(output_vec, input.shape.clone());
        } else {
            match &input[0].data {
                TensorData::F32(data) => {
                    let mut output_vec = vec![0.0; data.len()];

                    for i in 0..data.len() {
                        output_vec[i] = data[i].max(0.0);
                    }

                    return Tensor::new(output_vec, input[0].shape.clone());
                }
                TensorData::I8 { data, scales } => {
                    let mut output_vec = vec![0i8; data.len()];

                    for i in 0..data.len() {
                        output_vec[i] = data[i].max(0);
                    }

                    return Tensor::new_i8(output_vec, scales.clone(), input[0].shape.clone());
                }
            }
        }
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }

    fn prepare_inference(&mut self) {
        self.is_inference = true;
    }

    fn prepare_train(&mut self) {
        self.is_inference = false;
    }
}
