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
//! - `serialization`: Enable model saving/loading with serde (adds `serde`, `serde_json`, `anyhow`)
//! - `logging`: Enable TensorBoard logging (adds `tensorboard-rs`, `chrono`)
//! - `progress`: Enable progress bars during training (adds `indicatif`)
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
//! With serialization support:
//! ```toml
//! [dependencies]
//! minimum_ml = { version = "0.1.1", features = ["serialization"] }
//! ```
//!
//! With all features:
//! ```toml
//! [dependencies]
//! minimum_ml = { version = "0.1.1", features = ["full"] }
//! ```
//!
//! # Usage Example
//!
//! ```ignore
//! use minimum_ml::ml::{Tensor, Graph};
//! use minimum_ml::ml::params::MM;
//!
//! // Create a computational graph
//! let graph = Graph::new();
//!
//! // Create tensors and perform operations
//! let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
//! let linear = MM::new(3, 2);
//! let output = linear.forward(&graph, &x);
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
