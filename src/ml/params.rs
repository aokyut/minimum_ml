use super::xiver_vec;
use super::{Node, Result, Tensor, TensorData};
use crate::quantize::int_quantize::IntMM;


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
        let shape_size = input.shape.len();
        let batch = input.shape[shape_size - 2];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.height;

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
        let shape_size = input.shape.len();
        assert!(shape_size > 1);
        let batch = input.shape[shape_size - 2];
        let in_features = *input.shape.last().unwrap();
        assert_eq!(in_features, self.width);

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

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
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
        let shape_size = input.shape.len();
        let batch = input.shape[shape_size - 2];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.height;

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
        let shape_size = input.shape.len();
        assert!(shape_size > 1);
        let batch = input.shape[shape_size - 2];
        let in_features = *input.shape.last().unwrap();
        assert_eq!(in_features, self.width);

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

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
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
        let shape_size = input.shape.len();
        let batch = input.shape[shape_size - 2];
        let features = *input.shape.last().unwrap();
        assert_eq!(features, self.height);

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
        let shape_size = input.shape.len();
        assert!(shape_size > 1);
        let batch = input.shape[shape_size - 2];
        let features = *input.shape.last().unwrap();
        assert_eq!(features, self.height);

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

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct QuantizedLinear {
    pub w: Tensor,
    pub b: Tensor,
    pub height: usize,
    pub width: usize,
    pub w_grad: Option<Tensor>,
    pub b_grad: Option<Tensor>,
    pub int_mm: Option<IntMM>,
}

impl QuantizedLinear {
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
            int_mm: None,
        }
    }

    pub fn auto(input_size: usize, output_size: usize) -> Self {
        let w = xiver_vec(input_size, input_size * output_size);
        let w = Tensor::new(w, vec![output_size, input_size]);
        let b = Tensor::zeros(vec![output_size]);
        Self::new(w, b)
    }

    pub fn from_linear(linear: &Linear) -> Self {
        Self::new(linear.w.clone(), linear.b.clone())
    }

    pub fn prepare_inference_with_calib(&mut self, calibration_data: Option<&Tensor>) {
        let height = self.height;
        let width = self.width;
        let mut int_mm = IntMM::new(width, height);
        let w_f32 = self.w.as_f32_slice();
        let mut w_mat = Vec::with_capacity(height);
        for i in 0..height {
            w_mat.push(w_f32[i * width..(i + 1) * width].to_vec());
        }

        let mut xs = Vec::new();
        if let Some(calib) = calibration_data {
            let calib_f32 = calib.as_f32_slice();
            let n = calib.shape[0];
            for i in 0..n {
                xs.push(calib_f32[i * width..(i + 1) * width].to_vec());
            }
        }

        int_mm.train(&xs, &w_mat);
        self.int_mm = Some(int_mm);
    }
}

impl Node for QuantizedLinear {
    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];

        if let Some(int_mm) = &self.int_mm {
            match &input.data {
                TensorData::I8 { data, scales } => {
                    let batch_size = scales.len();
                    let mut out_data = Vec::with_capacity(batch_size * int_mm.d_out);
                    let mut out_scales = Vec::with_capacity(batch_size);
                    let b_f32 = self.b.as_f32_slice();

                    for i in 0..batch_size {
                        let row_i8 = &data[i * int_mm.d_in..(i + 1) * int_mm.d_in];
                        out_data.extend(int_mm.product_i8_i8(row_i8, scales[i], Some(&b_f32)));
                        out_scales.push(int_mm.output_scale);
                    }
                    let mut ans_shape = input.shape.clone();
                    *ans_shape.last_mut().unwrap() = int_mm.d_out;
                    return Tensor::new_i8(out_data, out_scales, ans_shape);
                }
                TensorData::F32(data) => {
                    let batch_size = data.len() / int_mm.d_in;
                    let mut out_data = Vec::with_capacity(batch_size * int_mm.d_out);
                    let b_f32 = self.b.as_f32_slice();
                    for i in 0..batch_size {
                        let row_f32 = &data[i * int_mm.d_in..(i + 1) * int_mm.d_in];
                        out_data.extend(int_mm.product_f32_i8(row_f32, Some(&b_f32)));
                    }
                    let mut ans_shape = input.shape.clone();
                    *ans_shape.last_mut().unwrap() = int_mm.d_out;
                    // Note: product_f32_i8 returns i8, but we might want f32 here or i8.
                    // Given the previous requirement "I8 in -> I8 out", we stick to i8 output.
                    let out_scales = vec![int_mm.output_scale; batch_size];
                    return Tensor::new_i8(out_data, out_scales, ans_shape);
                }
            }
        }

        // Fallback to F32 if IntMM is not prepared
        let input_f32 = input.as_f32_slice();
        let shape_size = input.shape.len();
        let batch = if shape_size > 1 {
            input.shape[shape_size - 2]
        } else {
            1
        };
        let in_features = *input.shape.last().unwrap();
        let out_features = self.w.shape[0];

        let mut ans_shape = input.shape.clone();
        *ans_shape.last_mut().unwrap() = out_features;
        let mut ans_data = vec![0.0; batch * out_features];
        let w_f32 = self.w.as_f32_slice();
        let b_f32 = self.b.as_f32_slice();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_ans = b * out_features;
            for i in 0..out_features {
                let mut sum = b_f32[i];
                for j in 0..in_features {
                    sum += input_f32[offset_input + j] * w_f32[i * in_features + j];
                }
                ans_data[offset_ans + i] = sum;
            }
        }
        Tensor::new(ans_data, ans_shape)
    }

    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let input = inputs[0];
        let shape_size = input.shape.len();
        let batch = input.shape[shape_size - 2];
        let in_features = *input.shape.last().unwrap();
        let out_features = self.height;

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

        // println!("input: {:#?}", input_data);
        // println!("w_data: {:#?}", w_data);
        // println!("grad_data: {:#?}", grad_data);
        // println!("w_grad: {:#?}", w_grad_f32);
        // println!("b_grad: {:#?}", b_grad_f32);
        // println!("input_grad: {:#?}", input_grad_f32);

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

    fn prepare_inference(&mut self) {
        self.prepare_inference_with_calib(None);
    }

    fn prepare_train(&mut self) {
        self.int_mm = None;
    }

    fn no_grad(&self) -> bool {
        false
    }
    fn has_params(&self) -> bool {
        true
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
}

#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct QuantizedMM {
    pub w: Tensor,
    pub int_mm: Option<IntMM>,
}

impl QuantizedMM {
    pub fn new(w: Tensor) -> Self {
        Self { w, int_mm: None }
    }

    pub fn from_mm(mm: &MM) -> Self {
        Self::new(mm.w.clone())
    }

    pub fn prepare_inference_with_calib(&mut self, calibration_data: Option<&Tensor>) {
        let height = self.w.shape[0];
        let width = self.w.shape[1];
        let mut int_mm = IntMM::new(width, height);
        let w_f32 = self.w.as_f32_slice();
        let mut w_mat = Vec::with_capacity(height);
        for i in 0..height {
            w_mat.push(w_f32[i * width..(i + 1) * width].to_vec());
        }

        let mut xs = Vec::new();
        if let Some(calib) = calibration_data {
            let calib_f32 = calib.as_f32_slice();
            let n = calib.shape[0];
            for i in 0..n {
                xs.push(calib_f32[i * width..(i + 1) * width].to_vec());
            }
        }

        int_mm.train(&xs, &w_mat);
        self.int_mm = Some(int_mm);
    }
}

impl Node for QuantizedMM {
    fn call(&self, input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(input_vec.len(), 1);
        let input = &input_vec[0];

        if let Some(int_mm) = &self.int_mm {
            match &input.data {
                TensorData::I8 { data, scales } => {
                    let batch_size = scales.len();
                    let mut out_data = Vec::with_capacity(batch_size * int_mm.d_out);
                    let mut out_scales = Vec::with_capacity(batch_size);

                    for i in 0..batch_size {
                        let row_i8 = &data[i * int_mm.d_in..(i + 1) * int_mm.d_in];
                        out_data.extend(int_mm.product_i8_i8(row_i8, scales[i], None));
                        out_scales.push(int_mm.output_scale);
                    }
                    let mut ans_shape = input.shape.clone();
                    *ans_shape.last_mut().unwrap() = int_mm.d_out;
                    return Tensor::new_i8(out_data, out_scales, ans_shape);
                }
                TensorData::F32(data) => {
                    let batch_size = data.len() / int_mm.d_in;
                    let mut out_data = Vec::with_capacity(batch_size * int_mm.d_out);
                    for i in 0..batch_size {
                        let row_f32 = &data[i * int_mm.d_in..(i + 1) * int_mm.d_in];
                        out_data.extend(int_mm.product_f32_i8(row_f32, None));
                    }
                    let mut ans_shape = input.shape.clone();
                    *ans_shape.last_mut().unwrap() = int_mm.d_out;
                    let out_scales = vec![int_mm.output_scale; batch_size];
                    return Tensor::new_i8(out_data, out_scales, ans_shape);
                }
            }
        }

        // Fallback to F32
        let input_f32 = input.as_f32_slice();
        let shape_size = input.shape.len();
        let batch = if shape_size > 1 {
            input.shape[shape_size - 2]
        } else {
            1
        };
        let in_features = *input.shape.last().unwrap();
        let out_features = self.w.shape[0];

        let mut ans_shape = input.shape.clone();
        *ans_shape.last_mut().unwrap() = out_features;
        let mut ans_data = vec![0.0; batch * out_features];
        let w_f32 = self.w.as_f32_slice();

        for b in 0..batch {
            let offset_input = b * in_features;
            let offset_ans = b * out_features;
            for i in 0..out_features {
                let mut sum = 0.0;
                for j in 0..in_features {
                    sum += input_f32[offset_input + j] * w_f32[i * in_features + j];
                }
                ans_data[offset_ans + i] = sum;
            }
        }
        Tensor::new(ans_data, ans_shape)
    }

    fn backward(&mut self, _grad: &Tensor, _inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        panic!("QuantizedMM does not support backward");
    }

    fn prepare_inference(&mut self) {
        self.prepare_inference_with_calib(None);
    }

    fn no_grad(&self) -> bool {
        false
    }
    fn has_params(&self) -> bool {
        false
    }
}
