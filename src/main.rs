#![allow(unused)]

use minimum_ml::ml;
use minimum_ml::quantize::{
    funcs::{Dequantize, QReLU, Quantize},
    params::QuantizedLinear,
};
use minimum_ml::utills::rand::get_random_normal;
use rand::Rng;

fn main() {
    train_mnist();
}

fn train_mnist() {
    let mut g = ml::Graph::new();

    let input = g.push_placeholder();
    let dequantized3 = minimum_ml::sequential!(
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
    let loss = g.add_layer(vec![dequantized3, target], Box::new(ml::funcs::MSE::new()));

    g.set_train_mode();
    g.set_inference_mode();
    g.set_train_mode();
    g.set_target(loss);

    // println!("g: {:#?}", g.layers.len());

    let optimizer = ml::optim::Adam::new(0.001, 0.9, 0.999);
    g.optimizer = Some(Box::new(optimizer));

    g.set_placeholder(vec![input, target]);

    println!("Loading MNIST...");
    let (train_images, train_labels) = load_mnist("mnist_train.txt");
    let n_train = train_images.len() / 784;
    println!("Loaded {} samples.", n_train);

    let mut loss_f32: Option<f32> = None;
    let mut rng = rand::rng();

    let batch = 64;
    for i in 0..10000 {
        let mut xs = Vec::with_capacity(batch * 784);
        let mut ys = Vec::with_capacity(batch * 10);
        for _ in 0..batch {
            let idx = rng.random_range(0..n_train);
            xs.extend_from_slice(&train_images[idx * 784..(idx + 1) * 784]);
            ys.extend_from_slice(&train_labels[idx * 10..(idx + 1) * 10]);
        }

        let target_tensor = ml::Tensor::new(ys, vec![batch, 10]);
        let input_tensor = ml::Tensor::new(xs, vec![batch, 784]);
        let result = g.forward(vec![input_tensor, target_tensor]);
        match loss_f32 {
            Some(loss) => {
                loss_f32 = Some(loss * 0.9 + 0.1 * result.as_f32_slice()[0]);
            }
            None => {
                loss_f32 = Some(result.as_f32_slice()[0]);
            }
        }
        if i % 100 == 0 {
            println!("[{i}]result: {:#?}", loss_f32.unwrap());
        }
        g.backward();
        g.optimize();
        g.reset();
    }

    println!("Evaluating...");
    let (test_images, test_labels) = load_mnist("mnist_test.txt");
    let n_test = test_images.len() / 784;
    let mut correct = 0;

    g.set_inference_mode();
    g.set_target(dequantized3);
    g.set_placeholder(vec![input]);

    for i in 0..1000 {
        let x = test_images[i * 784..(i + 1) * 784].to_vec();
        let y_true = &test_labels[i * 10..(i + 1) * 10];

        let input_tensor = ml::Tensor::new(x, vec![1, 784]);
        let result = g.forward(vec![input_tensor]);
        let res_slice = result.as_f32_slice();

        let mut max_idx = 0;
        let mut max_val = res_slice[0];
        for j in 1..10 {
            if res_slice[j] > max_val {
                max_val = res_slice[j];
                max_idx = j;
            }
        }

        let mut true_idx = 0;
        for j in 0..10 {
            if y_true[j] > 0.5 {
                true_idx = j;
                break;
            }
        }

        if max_idx == true_idx {
            correct += 1;
        }
        g.reset();
    }
    println!(
        "Test Accuracy: {}/1000 ({:.2}%)",
        correct,
        (correct as f32 / 10.0)
    );
}

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
    return x[0].sin() + x[1].cos();
}
