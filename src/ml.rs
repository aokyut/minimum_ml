/// Binary I/O utilities for model parameters.
pub(crate) mod binary_io;
/// Activation functions and loss functions.
pub mod funcs;
/// Logging utilities for training (requires `logging` feature).
#[cfg(feature = "logging")]
pub mod logger;
/// Metrics for model evaluation.
pub mod metrics;
/// Operations for tensor manipulation.
pub mod ops;
/// Optimization algorithms (SGD, Adam, etc.).
pub mod optim;
/// Neural network layer parameters and implementations.
pub mod params;
/// Progress tracking utilities.
pub mod progress;

/// Macro to create a sequential neural network layer composition.
///
/// # Example
/// ```ignore
/// let output_id = sequential!(graph, input_id, [
///     MM::new(784, 128),
///     ReLU {},
///     MM::new(128, 10),
/// ]);
/// ```
#[macro_export]
macro_rules! sequential {
    ($graph:expr, $input:expr, [$($node:expr),* $(,)?]) => {{
        let g = &mut $graph;
        let mut last_id = $input;
        $(
            last_id = g.add_layer(vec![last_id], Box::new($node));
        )*
        last_id
    }};
}

use crate::utills::rand::*;

pub type Result<T> = std::result::Result<T, std::io::Error>;

use std::borrow::Cow;
use std::fs;
use std::ops::{Add, AddAssign};
use std::path::PathBuf;

/// Creates a random vector using Xavier initialization.
///
/// # Arguments
/// * `n` - Input dimension for scaling
/// * `size` - Size of the output vector
pub fn xiver_vec(n: usize, size: usize) -> Vec<f32> {
    let sigma = (1.0 / n as f32).sqrt();

    get_random_normal(size, 0.0, sigma)
}

/// Data storage for tensors, supporting both f32 and quantized i8 formats.
#[derive(Clone, Debug)]
pub enum TensorData {
    F32(Vec<f32>),
    I8 {
        data: Vec<i8>,
        scales: Vec<f32>, // Per-row scales for batch processing
    },
}

/// Multi-dimensional array for neural network computations.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: TensorData,
    pub shape: Vec<usize>,
}

impl Tensor {
    pub fn from_shape(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let data = Tensor::create_random_array(size);
        Tensor {
            data: TensorData::F32(data),
            shape,
        }
    }

    pub fn zeros_like(tensor: &Tensor) -> Self {
        let size = tensor.len();
        match &tensor.data {
            TensorData::F32(_) => {
                let data = vec![0.0; size];
                Tensor {
                    data: TensorData::F32(data),
                    shape: tensor.shape.clone(),
                }
            }
            TensorData::I8 { scales, .. } => {
                // Return i8 zeros if it was i8, but usually zeros_like is for grads/accumulators
                // For simplicity, we fallback to f32 or return i8 with zero scales
                let data = vec![0i8; size];
                Tensor {
                    data: TensorData::I8 {
                        data,
                        scales: vec![0.0; scales.len()],
                    },
                    shape: tensor.shape.clone(),
                }
            }
        }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let data = vec![0.0; size];

        Tensor {
            data: TensorData::F32(data),
            shape,
        }
    }

    pub fn ones_like(tensor: &Tensor) -> Self {
        let size = tensor.len();
        let data = vec![1.0; size];

        Tensor {
            data: TensorData::F32(data),
            shape: tensor.shape.clone(),
        }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let mut size = 1;
        for s in shape.iter() {
            size *= s;
        }
        let data = vec![1.0; size];

        Tensor {
            data: TensorData::F32(data),
            shape,
        }
    }

    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for s in shape.iter() {
            size *= s;
        }
        assert_eq!(size, data.len());
        Tensor {
            data: TensorData::F32(data),
            shape,
        }
    }

    pub fn new_i8(data: Vec<i8>, scales: Vec<f32>, shape: Vec<usize>) -> Self {
        let mut size: usize = 1;
        for s in shape.iter() {
            size *= s;
        }
        assert_eq!(size, data.len());
        Tensor {
            data: TensorData::I8 { data, scales },
            shape,
        }
    }

    fn create_random_array(size: usize) -> Vec<f32> {
        get_random_normal(size, 0.0, 1.0)
    }

    pub fn null() -> Self {
        Tensor {
            data: TensorData::F32(Vec::new()),
            shape: Vec::new(),
        }
    }

    pub fn get_item(&self) -> Option<f32> {
        if self.len() == 1 {
            match &self.data {
                TensorData::F32(v) => Some(v[0]),
                TensorData::I8 { data, scales } => Some(data[0] as f32 / scales[0]),
            }
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        match &self.data {
            TensorData::F32(v) => v.len(),
            TensorData::I8 { data, .. } => data.len(),
        }
    }

    pub fn as_f32_slice(&self) -> Cow<'_, [f32]> {
        match &self.data {
            TensorData::F32(v) => Cow::Borrowed(v),
            TensorData::I8 { data, scales } => {
                let n_rows = scales.len();
                let row_dim = data.len() / n_rows;
                let mut out = Vec::with_capacity(data.len());
                for i in 0..n_rows {
                    let scale = scales[i];
                    let offset = i * row_dim;
                    for j in 0..row_dim {
                        out.push(data[offset + j] as f32 / scale);
                    }
                }
                Cow::Owned(out)
            }
        }
    }

    pub fn f32_data_mut(&mut self) -> &mut Vec<f32> {
        match &mut self.data {
            TensorData::F32(v) => v,
            _ => panic!("Tensor is not F32"),
        }
    }

    /// Returns the indices of the maximum values along a dimension.
    ///
    /// # Arguments
    /// * `dim` - If None, returns the index of the global maximum as a scalar tensor.
    ///          If Some(d), returns the indices of maximum values along dimension d.
    ///
    /// # Examples
    /// ```
    /// // Global argmax
    /// let t = Tensor::new(vec![1.0, 3.0, 2.0], vec![3]);
    /// let idx = t.argmax(None); // Returns Tensor with value 1.0 (index of max)
    ///
    /// // Argmax along last dimension (typical for classification)
    /// let t = Tensor::new(vec![0.1, 0.9, 0.3, 0.7], vec![2, 2]);
    /// let idx = t.argmax(Some(1)); // Returns [1.0, 1.0] (indices per row)
    /// ```
    pub fn argmax(&self, dim: Option<usize>) -> Tensor {
        let data = self.as_f32_slice();

        match dim {
            None => {
                // Global argmax
                let mut max_idx = 0;
                let mut max_val = data[0];
                for (i, &val) in data.iter().enumerate().skip(1) {
                    if val > max_val {
                        max_val = val;
                        max_idx = i;
                    }
                }
                Tensor::new(vec![max_idx as f32], vec![1])
            }
            Some(d) => {
                assert!(
                    d < self.shape.len(),
                    "Dimension {} out of bounds for shape {:?}",
                    d,
                    self.shape
                );

                // Calculate strides
                let mut strides = vec![1; self.shape.len()];
                for i in (0..self.shape.len() - 1).rev() {
                    strides[i] = strides[i + 1] * self.shape[i + 1];
                }

                let dim_size = self.shape[d];
                let _dim_stride = strides[d];

                // Calculate output shape (remove the dimension we're reducing over)
                let mut out_shape = self.shape.clone();
                out_shape.remove(d);
                if out_shape.is_empty() {
                    out_shape.push(1);
                }

                let out_size: usize = out_shape.iter().product();
                let mut result = vec![0.0; out_size];

                // Iterate over all possible index combinations
                for out_idx in 0..out_size {
                    // Convert flat output index to output coordinates
                    let mut coords = vec![0; self.shape.len()];
                    let mut temp_idx = out_idx;
                    let mut out_dim = 0;

                    for i in 0..self.shape.len() {
                        if i == d {
                            continue; // Skip the dimension we're reducing
                        }
                        let size = if out_dim < out_shape.len() {
                            out_shape[out_dim]
                        } else {
                            1
                        };
                        coords[i] = temp_idx % size;
                        temp_idx /= size;
                        out_dim += 1;
                    }

                    // Find max along dimension d
                    let mut max_idx = 0;
                    let mut max_val = f32::NEG_INFINITY;

                    for j in 0..dim_size {
                        coords[d] = j;

                        // Convert coordinates to flat index
                        let mut flat_idx = 0;
                        for (k, &coord) in coords.iter().enumerate() {
                            flat_idx += coord * strides[k];
                        }

                        let val = data[flat_idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = j;
                        }
                    }

                    result[out_idx] = max_idx as f32;
                }

                Tensor::new(result, out_shape)
            }
        }
    }
}

/// Creates a batched tensor from a vector of tensors.
///
/// # Arguments
/// * `tensors` - Vector of tensors to batch together
pub fn create_batch(tensors: Vec<Tensor>) -> Tensor {
    let size = tensors[0].len();
    let mut batch_data = Vec::new(); // Fallback to F32 for batch creation
    let mut shape = tensors[0].shape.clone();
    shape.insert(0, tensors.len());

    for tensor in tensors {
        assert_eq!(tensor.len(), size);
        batch_data.extend(tensor.as_f32_slice().iter());
    }

    Tensor::new(batch_data, shape)
}

impl Add for Tensor {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        assert_eq!(self.len(), rhs.len());

        let mut out_data = self.as_f32_slice().into_owned();
        let rhs_data = rhs.as_f32_slice();
        for i in 0..out_data.len() {
            out_data[i] += rhs_data[i];
        }

        Tensor::new(out_data, self.shape.clone())
    }
}

impl AddAssign for Tensor {
    fn add_assign(&mut self, rhs: Self) {
        let size = self.len();
        assert_eq!(size, rhs.len());

        let rhs_data = rhs.as_f32_slice();
        match &mut self.data {
            TensorData::F32(v) => {
                for i in 0..size {
                    v[i] += rhs_data[i];
                }
            }
            _ => panic!("AddAssign only supported for F32 target"),
        }
    }
}

/// Trait for neural network layers and operations.
///
/// Implementors define forward and backward passes for automatic differentiation.
pub trait Node {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, output: &Tensor) -> Vec<Tensor>;
    fn call(&self, input: Vec<Tensor>) -> Tensor;
    fn no_grad(&self) -> bool {
        false
    }
    fn has_params(&self) -> bool {
        false
    }
    fn apply_update(&mut self, _update: Vec<Tensor>) {}
    fn load_param(&mut self, _file: PathBuf) -> Result<()> {
        Ok(())
    }
    fn save_param(&self, _file: PathBuf) -> Result<()> {
        Ok(())
    }
    fn pull_grad(&self) -> Option<Vec<&Tensor>> {
        None
    }
    fn print(&self) {}
    fn prepare_inference(&mut self) {}
    fn prepare_train(&mut self) {}
}

/// Trait for optimization algorithms.
pub trait Optimizer {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor>;
}

/// Placeholder node for graph inputs.
pub struct Placeholder {}

impl Node for Placeholder {
    fn backward(&mut self, _: &Tensor, _: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        vec![]
    }
    fn call(&self, _: Vec<Tensor>) -> Tensor {
        Tensor::null()
    }
    fn no_grad(&self) -> bool {
        true
    }
}

impl Default for Placeholder {
    fn default() -> Self {
        Self::new()
    }
}

impl Placeholder {
    pub fn new() -> Self {
        Placeholder {}
    }
}

/// Computational graph for automatic differentiation.
///
/// Manages forward and backward passes through a network of nodes.
pub struct Graph {
    pub layers: Vec<Box<dyn Node>>,
    pub optimizer: Option<Box<dyn Optimizer>>,
    pub flows: Vec<Option<Tensor>>,
    pub backflows: Vec<Option<Tensor>>,
    placeholder: Option<Vec<usize>>,
    parameters: Vec<usize>,
    inputs: Vec<Vec<usize>>,
    output: Vec<usize>,
    pub target: usize,
    pub is_inference: bool,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            layers: Vec::new(),
            flows: Vec::new(),
            optimizer: None,
            backflows: Vec::new(),
            parameters: Vec::new(),
            placeholder: None,
            inputs: Vec::new(),
            output: Vec::new(),
            target: 0,
            is_inference: false,
        }
    }

    pub fn set_inference_mode(&mut self) {
        self.is_inference = true;
        for layer in self.layers.iter_mut() {
            layer.prepare_inference();
        }
    }

    pub fn set_train_mode(&mut self) {
        self.is_inference = false;
        for layer in self.layers.iter_mut() {
            layer.prepare_train();
        }
    }

    pub fn backward(&mut self) {
        let mut stack: Vec<usize> = vec![self.target];
        self.backflows[self.target] =
            Some(Tensor::ones_like(self.flows[self.target].as_ref().unwrap()));

        while let Some(tar) = stack.pop() {
            let input_ids = &self.inputs[tar];
            let mut input_vecs = Vec::new();
            if self.layers[tar].no_grad() {
                continue;
            }
            for input_id in input_ids.iter() {
                input_vecs.push(self.flows[*input_id].as_ref().unwrap());
                stack.push(*input_id);
            }
            let mut input_grads = self.layers[tar].backward(
                self.backflows[tar].as_ref().unwrap(),
                input_vecs,
                self.flows[tar].as_ref().unwrap(),
            );

            for i in 0..input_ids.len() {
                let input_id = input_ids[i];
                let swap = std::mem::replace(&mut input_grads[i], Tensor::null());
                if let Some(existing_grad) = self.backflows[input_id].as_mut() {
                    *existing_grad += swap;
                } else {
                    self.backflows[input_id] = Some(swap);
                }
            }
            // println!("tar: {}, input: {:#?}", tar, input_ids);
        }
    }

    pub fn inference(&self, mut input_vec: Vec<Tensor>) -> Tensor {
        let placeholder = self.placeholder.as_ref().unwrap();
        assert_eq!(placeholder.len(), input_vec.len());
        let mut flows = vec![None; self.layers.len()];

        // println!("placeholder:{placeholder:?}");
        for i in 0..placeholder.len() {
            let id = placeholder[i];

            let input = std::mem::replace(&mut input_vec[i], Tensor::null());
            flows[id] = Some(input);
        }

        let mut stack: Vec<usize> = vec![self.target];
        while let Some(tar) = stack.pop() {
            stack.push(tar);
            let input_ids = &self.inputs[tar];
            let mut full = true;
            for input_id in input_ids.iter() {
                if flows[*input_id].is_none() {
                    stack.push(*input_id);
                    full = false;
                }
            }
            if !full {
                continue;
            }
            stack.pop();
            let mut inputs = Vec::new();
            for input_id in input_ids.iter() {
                let input = flows[*input_id].clone();
                inputs.push(input.unwrap());
            }
            let out = self.layers[tar].call(inputs);
            flows[tar] = Some(out);
        }
        let output = flows[self.target].clone().unwrap();
        // println!("{flows:?}");
        output.clone()
    }

    pub fn forward(&mut self, input_vec: Vec<Tensor>) -> Tensor {
        // println!("[g]input:{:?}", input_vec);
        self.forward_(self.placeholder.as_ref().unwrap().clone(), input_vec)
    }

    pub fn forward_(&mut self, placeholder: Vec<usize>, mut input_vec: Vec<Tensor>) -> Tensor {
        assert_eq!(placeholder.len(), input_vec.len());
        for i in 0..placeholder.len() {
            let id = placeholder[i];

            let input = std::mem::replace(&mut input_vec[i], Tensor::null());
            self.flows[id] = Some(input);
        }

        let mut stack: Vec<usize> = vec![self.target];
        while let Some(tar) = stack.pop() {
            stack.push(tar);
            let input_ids = &self.inputs[tar];
            let mut full = true;
            for input_id in input_ids.iter() {
                if self.flows[*input_id].is_none() {
                    stack.push(*input_id);
                    full = false;
                }
            }
            if !full {
                continue;
            }
            stack.pop();
            let mut inputs = Vec::new();
            for input_id in input_ids.iter() {
                let input = self.flows[*input_id].clone();
                inputs.push(input.unwrap());
            }
            let out = self.layers[tar].call(inputs);
            self.flows[tar] = Some(out);
            // println!("flows: {:#?}", self.flows);
        }

        self.flows[self.target].clone().unwrap()
    }

    pub fn optimize(&mut self) {
        if let Some(optimizer) = self.optimizer.as_mut() {
            //TODO
            for &id in self.parameters.iter() {
                // println!("id: {}", id);
                let grads = self.layers[id].pull_grad().unwrap();
                // println!("grads: {:#?}", grads);
                let update = optimizer.optimize(id, grads);
                self.layers[id].apply_update(update);
            }
        }
    }

    pub fn push_placeholder(&mut self) -> usize {
        let placeholder = Placeholder {};
        self.layers.push(Box::new(placeholder));
        self.flows.push(None);
        self.backflows.push(None);
        self.inputs.push(Vec::new());

        let id = self.inputs.len() - 1;
        self.output.push(id);

        id
    }

    pub fn add_layer(&mut self, inputs: Vec<usize>, node: Box<dyn Node>) -> usize {
        let has_param = node.has_params();
        self.layers.push(node);
        self.flows.push(None);
        self.backflows.push(None);

        let id = self.layers.len() - 1;
        self.output.push(id);
        if has_param {
            self.parameters.push(id);
        }

        for i in inputs.iter() {
            self.output[*i] = id;
        }

        self.inputs.push(inputs);

        id
    }

    pub fn reset(&mut self) {
        for i in 0..self.flows.len() {
            self.flows[i] = None;
            self.backflows[i] = None
        }
    }

    pub fn set_target(&mut self, id: usize) {
        self.target = id;
    }

    pub fn set_placeholder(&mut self, placeholder: Vec<usize>) {
        self.placeholder = Some(placeholder);
    }

    pub fn save(&self, file: &str) {
        let mut path = PathBuf::new();
        path.push(file);

        let _ = fs::create_dir_all(path.clone());

        for i in 0..self.layers.len() {
            path.push(format!("{}.param", i));

            let _ = self.layers[i].save_param(path.clone());
            path = path.parent().unwrap().to_path_buf();
        }
    }

    pub fn load(&mut self, file: String) {
        let mut path = PathBuf::new();
        path.push(file);

        for i in 0..self.layers.len() {
            path.push(format!("{}.param", i));

            let _ = self.layers[i].load_param(path.clone());
            path = path.parent().unwrap().to_path_buf();
        }
    }
}
