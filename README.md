# minimum_ml

**Experimental Machine Learning Library in Rust**

`minimum_ml` is a lightweight, experimental machine learning library designed for educational purposes and minimal dependency environments. It provides basic building blocks for neural networks with a focus on simplicity and low footprint.

## Features

- **Minimal Dependencies**: The core library depends only on `getrandom` (for seeding) and a local derive macro. No heavyweight crates like `ndarray`, `tch`, or `rand` are required by default.
- **Custom RNG**: Uses a custom `XorShift64` implementation for random number generation.
- **Quantization Support**: Includes 8-bit integer (i8) quantization layers and SIMD-optimized (AVX2) matrix multiplication.
- **Lightweight Logging**: Built-in TensorBoard-compatible logger (Scalars only) using pure Rust.
- **Automatic Differentiation**: Basic backward propagation engine.

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
minimum_ml = { git = "https://github.com/aokyut/minimum_ml" }
```

### Features

- `default`: No extra features.
- `logging`: Enables TensorBoard logging support (std-based, no external deps).
- `full`: Enables all features.

## Usage Guide

### 1. Defining a Network with `sequential!`

You can define a neural network layer-by-layer using the `sequential!` macro. This handles the connections between layers automatically.

```rust
use minimum_ml::ml::{Graph, Tensor};
use minimum_ml::ml::params::MM;
use minimum_ml::ml::funcs::{ReLU, Softmax, CrossEntropyLoss};

fn main() {
    let mut g = Graph::new();
    
    // 1. Define placeholders for input and target
    let input = g.push_placeholder();
    let target = g.push_placeholder();
    
    // 2. Define the network structure
    let network_output = minimum_ml::sequential!(
        g,
        input,
        [
            MM::new(784, 128), // Linear layer (784 -> 128)
            ReLU::new(),       // Activation
            MM::new(128, 10),  // Linear layer (128 -> 10)
            Softmax::new(),    // Output activation
        ]
    );

    // 3. Define Loss
    // Connect network output and target to the loss function
    let loss = g.add_layer(
        vec![network_output, target], 
        Box::new(CrossEntropyLoss::new())
    );
    
    // 4. Set Graph Targets
    // Tell the graph what to compute (loss) and what inputs to expect
    g.set_target(loss);
    g.set_placeholder(vec![input, target]);
    
    // (Optional) Set Optimizer
    let optimizer = minimum_ml::ml::optim::Adam::new(0.001, 0.9, 0.999);
    g.optimizer = Some(Box::new(optimizer));
}
```

### 2. Training Loop (Forward & Backward)

Once the graph is set up, you can run the training loop:

```rust
// Inside the training loop...
// let input_tensor = ...;
// let target_tensor = ...;

// Forward Pass
let loss_val = g.forward(vec![input_tensor, target_tensor]);
println!("Loss: {:?}", loss_val.as_f32_slice()[0]);

// Backward Pass (Gradient Calculation)
g.backward();

// Optimization (Update Weights)
g.optimize();

// Reset Gradients/Flows for next iteration
g.reset();
```

### 3. Using Datasets and Dataloader

Implement the `Dataset` trait for your data, then use `Dataloader` for batching and shuffling.

```rust
use minimum_ml::dataset::{Dataset, Dataloader, Stackable};

// 1. Define your data item structure
#[derive(Stackable)] // Macro to help batching
pub struct MyData {
    x: Tensor,
    y: Tensor,
}

// 2. Define your Dataset struct
struct MyDataset {
    data: Vec<MyData>,
}

// 3. Implement Dataset trait
impl Dataset for MyDataset {
    type Item = MyData;
    fn len(&self) -> usize { self.data.len() }
    fn get(&self, index: usize) -> Self::Item { 
        // Return data item at index (cloning is typical here)
        // ... 
    }
}

// 4. Use Dataloader
let dataset = MyDataset { ... };
let loader = Dataloader::new(dataset, 64 /* batch_size */, true /* strict_batch_size */);

for batch in loader.iter_batch() {
    let batch_x = batch.x; // Batched Tensor
    let batch_y = batch.y;
    // ... feed to graph ...
}
```

### 4. Saving and Loading Models

You can save and load the trained parameters.

```rust
// Save model
// Creates a directory "my_model" and saves parameters inside
g.save("my_model");

// Load model
// Loads parameters from "my_model" directory
g.load("my_model");
```

### 5. Available Components

| Component | Description |
|-----------|-------------|
| `ml.params.MM` | Matrix Multiplication (Linear Layer) |
| `ml.params.Bias` | Bias addition layer |
| `ml.params.Linear` | Combined MM + Bias (Helper) |
| `ml.funcs.ReLU` | ReLU Activation |
| `ml.funcs.Sigmoid` | Sigmoid Activation |
| `ml.funcs.Softmax` | Softmax Activation |
| `ml.funcs.CrossEntropyLoss` | Cross Entropy Loss function |
| `ml.funcs.MSELoss` | Mean Squared Error Loss function |
| `quantize.funcs.Quantize` | Quantize when inference mode |
| `quantize.funcs.Dequantize` | Dequantize when inference mode |
| `quantize.funcs.QReLU` | Quantized ReLU |
| `quantize.params.QuantizedLinear` | Linear layer for int8 quantization |
| `quantize.params.QuantizedMM` | Matrix Multiplication for int8 quantization |

## Running Tests

To run the test suite:

```bash
cargo test --lib
```

## License

MIT
