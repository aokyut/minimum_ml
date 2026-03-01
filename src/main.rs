#![allow(unused)]
#![allow(missing_docs)]

use minimum_ml::dataset::{Dataloader, Dataset, Stackable};
use minimum_ml::{ml, sequential};
use minimum_ml::utills::rand::get_random_normal;
use minimum_ml::{
    ml::Tensor,
    quantize::{
        funcs::{Dequantize, QReLU, Quantize},
        params::QuantizedLinear,
    },
};


fn main() {
    // train_mnist();
    train_ordinal_model();
}

#[derive(Stackable)]
pub struct MnistData {
    x: Tensor,
    y: Tensor,
}

pub struct MnistDataset {
    x_data: Vec<f32>,
    y_data: Vec<f32>,
}

impl MnistDataset {
    fn new(file: &str) -> Self {
        let (train_images, train_labels) = load_mnist(file);
        Self {
            x_data: train_images,
            y_data: train_labels,
        }
    }
}

impl Dataset for MnistDataset {
    type Item = MnistData;
    fn len(&self) -> usize {
        self.y_data.len() / 10
    }

    fn get(&self, index: usize) -> Self::Item {
        let x_tensor = Tensor::new(
            self.x_data[index * 784..(index + 1) * 784].to_vec(),
            vec![784],
        );
        let y_tensor = Tensor::new(self.y_data[index * 10..(index + 1) * 10].to_vec(), vec![10]);
        MnistData {
            x: x_tensor,
            y: y_tensor,
        }
    }
}

fn train_mnist() {
    use minimum_ml::ml::logger::TensorBoardLogger;
    let mut g = ml::Graph::new();

    let input = g.push_placeholder();
    let dequantized = minimum_ml::sequential!(
        g,
        input,
        [
            Quantize::new(),
            QuantizedLinear::auto(784, 64),
            // Dequantize::new(),
            QReLU::new(),
            // Quantize::new(),
            QuantizedLinear::auto(64, 10),
            Dequantize::new(),
            ml::funcs::Softmax::new(),
        ]
    );

    let target = g.push_placeholder();
    let loss = g.add_layer(vec![dequantized, target], Box::new(ml::funcs::CrossEntropyLoss::new()));

    g.set_train_mode();
    g.set_target(loss);

    // println!("g: {:#?}", g.layers.len());

    let optimizer = ml::optim::Adam::new(0.001, 0.9, 0.999);
    g.optimizer = Some(Box::new(optimizer));

    g.set_placeholder(vec![input, target]);

    let batch_size = 64;
    let epoch = 20;
    println!("Loading MNIST...");
    let train_dataloader = Dataloader::new(MnistDataset::new("MNIST_train.txt"), batch_size, true);
    println!("Loaded {} samples.", train_dataloader.len());
    println!("Loading MNIST test...");
    let test_dataloader = Dataloader::new(MnistDataset::new("MNIST_test.txt"), batch_size, false);
    println!("Loaded {} samples.", test_dataloader.len());

    let mut loss_f32: Option<f32> = None;


    let mut logger = TensorBoardLogger::new().unwrap();

    let mut step = 0;
    for e in 0..epoch {
        for batch in train_dataloader.iter_batch() {
            step += 1;
            let input_tensor = batch.x;
            let target_tensor = batch.y;
            let result = g.forward(vec![input_tensor.clone(), target_tensor.clone()]);
            match loss_f32 {
                Some(loss) => {
                    loss_f32 = Some(loss * 0.9 + result.as_f32_slice()[0] * 0.1);
                }
                None => {
                    loss_f32 = Some(result.as_f32_slice()[0]);
                }
            }

            logger.log_scalar("train/loss", loss_f32.unwrap()).unwrap();
            
            g.backward();
            g.optimize();
            g.reset();
            
            if step % 100 == 0 {
                g.set_inference_mode();
                g.set_target(dequantized);
                g.set_placeholder(vec![input]);
                let result = g.forward(vec![input_tensor.clone()]);
                let accuracy = ml::metrics::accuracy(&result, &target_tensor);
                logger.log_scalar("train/accuracy", accuracy).unwrap();
                
                println!("[{step}]loss: {:#?}, accuracy: {:#?}", loss_f32.unwrap(), accuracy);
                
                g.set_train_mode();
                g.set_target(loss);
                g.set_placeholder(vec![input, target]);
            }
            logger.next_step();

        }
        
        // Evaluate on test set after each epoch
        // println!("\nEpoch {} - Evaluating...", e + 1);
        g.set_inference_mode();
        g.set_target(dequantized);
        g.set_placeholder(vec![input]);
        
        let mut total_acc = 0.0;
        let mut n_batches = 0;
        
        for batch in test_dataloader.iter_batch() {
            let input_tensor = batch.x;
            let target_tensor = batch.y;
            let result = g.forward(vec![input_tensor]);
            
            // Calculate accuracy for this batch
            let acc = ml::metrics::accuracy(&result, &target_tensor);
            total_acc += acc;
            n_batches += 1;
            
            g.reset();
        }
        
        let avg_acc = total_acc / n_batches as f32;
        println!("Epoch {} - Test Accuracy: {:.2}%", e + 1, avg_acc * 100.0);
        logger.log_scalar("eval/accuracy", avg_acc).unwrap();   

        // Switch back to training mode
        g.set_train_mode();
        g.set_target(loss);
        g.set_placeholder(vec![input, target]);
    }

    // Final evaluation
    println!("\n=== Final Evaluation ===");
    g.set_inference_mode();
    g.set_target(dequantized);
    g.set_placeholder(vec![input]);
    
    let mut total_acc = 0.0;
    let mut n_batches = 0;
    
    for batch in test_dataloader.iter_batch() {
        let input_tensor = batch.x;
        let target_tensor = batch.y;
        let result = g.forward(vec![input_tensor]);
        
        let acc = ml::metrics::accuracy(&result, &target_tensor);
        total_acc += acc;
        n_batches += 1;
        
        g.reset();
    }
    
    let final_acc = total_acc / n_batches as f32;
    g.save("model");
    println!("Final Test Accuracy: {:.2}%\n", final_acc * 100.0);
}

/// Load MNIST-txt dataset file
fn load_mnist(path: &str) -> (Vec<f32>, Vec<f32>) {
    use std::io::BufRead;
    let file = std::fs::File::open(path).expect("Failed to open MNIST file");
    let reader = std::io::BufReader::new(file);
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for line in reader.lines() {
        let line = line.unwrap();
        let mut parts = line.split(",");
        if let Some(label_str) = parts.next() {
            let label: usize = label_str.parse().unwrap();
            let mut one_hot = vec![0.0; 10];
            one_hot[label] = 1.0;
            labels.extend(one_hot);

            for pixel_str in parts {
                let pixel: f32 = pixel_str.parse().unwrap();
                images.push(pixel / 255.0);
            }
        }
    }
    (images, labels)
}

fn test_func(x: Vec<f32>) -> f32 {
    x[0].sin() + x[1].cos()
}

fn train_ordinal_model() {
    use minimum_ml::ml::{self, funcs, params};
    let mut g = ml::Graph::new();
    let input = g.push_placeholder();
    let target = g.push_placeholder();
    let target_mask = g.push_placeholder();

    let optim = minimum_ml::ml::optim::Adam::new(0.001, 0.9, 0.999);
    g.set_optimizer(optim);

    // [Batch, 16]
    let output = minimum_ml::sequential!(
        g,
        input,
        [
            ml::params::Linear::auto(96, 512),
            ml::funcs::ReLU::new(),
            ml::params::Linear::auto(512, 256),
            ml::funcs::ReLU::new(),
            ml::params::Linear::auto(256, 1),
            ml::funcs::Tile::new(vec![1, 16]),
        ]
    );
    let param_tensor = Tensor::ones(vec![1, 16]);
    let param = params::Parameter::new(param_tensor);
    let param = g.add_parameter(param);
    let th = g.add_layer(vec![param], Box::new(funcs::CumlativeSum::new(1)));
    let th = sequential!{
        g, param,
        [
            funcs::CumlativeSum::new(1),
            funcs::SoftPlus::new(),
        ]
    };
    let diff = g.add_layer(vec![output, th], Box::new(funcs::Sub::new()));
    let probs = g.add_layer(vec![diff], Box::new(funcs::Sigmoid::new(1.0)));

    let bce = g.add_layer(vec![probs, target], Box::new(funcs::BinaryCrossEntropy::new()));
    let bce_masked = g.add_layer(vec![bce, target_mask], Box::new(funcs::Mul::new()));
    let sum = g.add_layer(vec![bce_masked], Box::new(funcs::Sum::new(Some(1), false)));
    let loss = g.add_layer(vec![sum], Box::new(funcs::Mean::new(None, false)));

    g.set_target(loss);
    g.set_train_mode();
    g.set_placeholder(vec![input, target, target_mask]);

    let input_tensor = Tensor::ones(vec![64, 96]);
    let target_tensor = Tensor::ones(vec![64, 16]);
    let target_mask_tensor = Tensor::ones(vec![64, 16]);

    let result = g.forward(vec![input_tensor, target_tensor, target_mask_tensor]);

    println!("result:{:#?}", result);

    g.backward();
    g.optimize();
    g.reset();

    
}