#![allow(missing_docs)]
//! Experimental Machine Learning Library
//!
//! `minimum_ml` is a lightweight, experimental machine learning library written in Rust.
//! It provides basic building blocks for neural networks, including:
//!
//! - Tensor operations and automatic differentiation
//! - Common activation functions (ReLU, Sigmoid, Softmax, etc.)
//! - Neural network layers (Linear, Convolution, etc.)
//! - Optimizers (SGD, Adam)
//! - Quantization support for model compression
//! - Dataset utilities for batching and iteration
//!
//! # Features
//!
//! This crate supports optional features to minimize dependencies:
//!
//! - `logging`: Enable TensorBoard logging (std-based, no external deps)
//! - `full`: Enable all features
//!
//! By default, **no optional features are enabled**, keeping the dependency footprint minimal.
//!
//! ## Examples
//!
//! Minimal install (only core dependencies):
//! ```toml
//! [dependencies]
//! minimum_ml = "0.1.1"
//! ```
//!
//! # Usage Example
//!
//! Using the `sequential!` macro to define a network:
//!
//! ```
//! use minimum_ml::ml::{Tensor, Graph};
//! use minimum_ml::ml::params::MM;
//! use minimum_ml::ml::funcs::{ReLU, Softmax, CrossEntropyLoss};
//!
//! fn main() {
//!     let mut g = Graph::new();
//!     
//!     // 1. Define placeholders for input and target
//!     let input = g.push_placeholder();
//!     let target = g.push_placeholder();
//!     
//!     // 2. Define the network structure
//!     let network_output = minimum_ml::sequential!(
//!         g,
//!         input,
//!         [
//!             MM::new(784, 128), // Linear layer (784 -> 128)
//!             ReLU::new(),       // Activation
//!             MM::new(128, 10),  // Linear layer (128 -> 10)
//!             Softmax::new(),    // Output activation
//!         ]
//!     );
//!
//!     // 3. Define Loss
//!     let loss = g.add_layer(
//!         vec![network_output, target], 
//!         Box::new(CrossEntropyLoss::new())
//!     );
//!     
//!     // 4. Set Graph Targets
//!     g.set_target(loss);
//!     g.set_placeholder(vec![input, target]);
//! }
//! ```

/// Dataset utilities for loading and batching data.
pub mod dataset;
/// Core machine learning primitives including tensors, graphs, and layers.
pub mod ml;
/// Quantization utilities for model compression and acceleration.
pub mod quantize;
/// Utility functions for random number generation and other helpers.
pub mod utills;

#[cfg(test)]
mod test;

pub fn add(left: u64, right: u64) -> u64 {
    left + right
}
