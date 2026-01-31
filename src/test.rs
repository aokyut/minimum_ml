use crate::ml::{self, Node, Tensor};

#[test]
fn test_linear_benchmark() {
    use crate::ml::funcs::QuantizeNode;
    use crate::utills::rand::get_random_normal;
    use std::time::Instant;

    let d_in = 256;
    let d_out = 128;
    let n_samples = 1000;

    // 1. レイヤーとデータの準備
    let linear = ml::params::Linear::auto(d_in, d_out);
    let xs_f32_raw = (0..n_samples)
        .map(|_| get_random_normal(d_in, 0.0, 1.0))
        .collect::<Vec<_>>();
    let mut input_data = Vec::with_capacity(n_samples * d_in);
    for x in &xs_f32_raw {
        input_data.extend(x);
    }
    let input_tensor = Tensor::new(input_data, vec![n_samples, d_in]);
    let input_f32 = input_tensor.clone();

    // 2. Step 1: Standard F32 Path (Baseline)
    let start_f32 = Instant::now();
    let out_f32 = linear.call(vec![input_f32]);
    let dur_f32 = start_f32.elapsed();
    let baseline_data = out_f32.as_f32_slice().to_vec();

    // 3. Step 2: quantize Mode setup (using specialized layers)
    let mut q_linear = ml::params::QuantizedLinear::from_linear(&linear);
    // q_linear.prepare_inference_with_calib(Some(&input_tensor));
    q_linear.prepare_inference();
    let mut q_node = QuantizeNode::new();
    q_node.prepare_inference();
    let dq_node = ml::funcs::DequantizeNode::new();

    // 4. Step 3: Online quantize of Input
    let start_quant = Instant::now();
    let input_i8 = q_node.call(vec![input_tensor]);
    let dur_quant = start_quant.elapsed();

    // 5. Step 4: Optimized I8 Path (now I8 -> I8)
    let start_i8 = Instant::now();
    let out_i8 = q_linear.call(vec![input_i8]);
    let dur_i8 = start_i8.elapsed();

    // 6. Step 5: Dequantize
    let start_dequant = Instant::now();
    let out_f32_final = dq_node.call(vec![out_i8]);
    let dur_dequant = start_dequant.elapsed();
    let quantized_data = out_f32_final.as_f32_slice().to_vec();

    // 7. Step 6: Evaluation
    println!(
        "\n=== Linear Layer Benchmark (In={}, Out={}, N={}) ===",
        d_in, d_out, n_samples
    );
    println!(
        "F32 Path (Baseline):      {:.2} ns/sample",
        dur_f32.as_nanos() as f64 / n_samples as f64
    );
    println!(
        "Input quantize:       {:.2} ns/sample",
        dur_quant.as_nanos() as f64 / n_samples as f64
    );
    println!(
        "I8 Path (Optimized):      {:.2} ns/sample",
        dur_i8.as_nanos() as f64 / n_samples as f64
    );
    println!(
        "Output Dequantize:    {:.2} ns/sample",
        dur_dequant.as_nanos() as f64 / n_samples as f64
    );
    let total_dur = dur_quant + dur_i8 + dur_dequant;
    println!(
        "Total I8 (Q + Exec + DQ): {:.2} ns/sample",
        total_dur.as_nanos() as f64 / n_samples as f64
    );
    println!(
        "Speedup (Exec Only):      {:.2}x",
        dur_f32.as_nanos() as f64 / dur_i8.as_nanos() as f64
    );
    println!(
        "Speedup (Total):          {:.2}x",
        dur_f32.as_nanos() as f64 / total_dur.as_nanos() as f64
    );

    evaluate_metrics(
        "Linear quantize Accuracy",
        &baseline_data,
        &quantized_data,
        total_dur.as_nanos(),
    );
}

#[test]
fn test_mat_mul() {
    let w = ml::Tensor::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
    let x = ml::Tensor::new(vec![0.0, 1.0, 2.0], vec![1, 3]);

    let mm = ml::params::MM::new(w);
    let out = mm.call(vec![x]);

    let expected = Tensor::new(vec![5.0, 14.0], vec![1, 2]);

    assert_tensor(expected, out, String::new());
}

fn evaluate_metrics(name: &str, exacts: &[f32], approxs: &[f32], duration_ns: u128) {
    let n = exacts.len();
    if n == 0 {
        return;
    }

    let mut sum_abs_diff = 0.0;
    let mut sum_sq_diff = 0.0;
    let mut sum_abs_exact = 0.0;

    let sum_exact: f32 = exacts.iter().sum();
    let mean_exact = sum_exact / n as f32;
    let mut sum_sq_variance = 0.0;

    for (e, a) in exacts.iter().zip(approxs.iter()) {
        let diff = e - a;
        sum_abs_diff += diff.abs();
        sum_sq_diff += diff * diff;
        sum_abs_exact += e.abs();
        sum_sq_variance += (e - mean_exact).powi(2);
    }

    let mae = sum_abs_diff / n as f32;
    let mse = sum_sq_diff / n as f32;
    let rmse = mse.sqrt();
    let avg_abs_exact = sum_abs_exact / n as f32;
    let relative_error = if avg_abs_exact > 0.0 {
        mae / avg_abs_exact
    } else {
        0.0
    };
    let r2 = if sum_sq_variance > 0.0 {
        1.0 - sum_sq_diff / sum_sq_variance
    } else {
        0.0
    };

    println!("\n=== {} Evaluation (n={}) ===", name, n);
    println!("MAE:            {:.6}", mae);
    println!("RMSE:           {:.6}", rmse);
    println!(
        "Relative Error: {:.6} ({:.2}%)",
        relative_error,
        relative_error * 100.0
    );
    println!("R2 Score:       {:.6}", r2);
    println!("Avg Exact Abs:  {:.6}", avg_abs_exact);
    println!("Avg Time/Query: {:.2} ns", duration_ns as f64 / n as f64);
    println!("====================================\n");
}

fn generate_structured_data(n: usize, d: usize, latent_d: usize) -> Vec<Vec<f32>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    // ランダムな投影行列 [latent_d, d]
    let mut projection = vec![vec![0.0f32; d]; latent_d];
    for i in 0..latent_d {
        for j in 0..d {
            projection[i][j] = rng.random::<f32>() - 0.5;
        }
    }

    let mut xs = Vec::with_capacity(n);
    for _ in 0..n {
        // 低次元の潜在ベクトルを生成
        let latent: Vec<f32> = (0..latent_d).map(|_| rng.random::<f32>() - 0.5).collect();

        // 高次元に投影
        let mut x = vec![0.0f32; d];
        for i in 0..latent_d {
            for j in 0..d {
                x[j] += latent[i] * projection[i][j];
            }
        }

        // ノイズを少し加える
        for j in 0..d {
            x[j] += (rng.random::<f32>() - 0.5) * 0.05;
        }

        xs.push(x);
    }
    xs
}

#[test]
fn test_int_mm() {
    use crate::quantize::int_quantize::IntMM;
    use crate::utills::rand::get_random_normal;
    use std::time::Instant;

    let d = 256;
    let n = 10000;
    let test_n = 1000;
    let d_out = 1;

    // 1. i8 vs u8 精度比較用データの生成 (Standard Normal)
    let w = get_random_normal(d * d_out, 0.0, 1.0);
    let xs = (0..n)
        .map(|_| get_random_normal(d, 0.0, 1.0))
        .collect::<Vec<_>>();

    // i8 (Symmetric)
    let max_w = w.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let scale_i8 = 127.0 / max_w;
    let w_i8: Vec<i8> = w.iter().map(|&v| (v * scale_i8).round() as i8).collect();
    let err_i8: f32 = w
        .iter()
        .zip(w_i8.iter())
        .map(|(&v, &q)| (v - (q as f32 / scale_i8)).powi(2))
        .sum::<f32>()
        .sqrt();

    // u8 (Min-Max)
    let min_w = w.iter().fold(f32::MAX, |a, &b| a.min(b));
    let max_w_raw = w.iter().fold(f32::MIN, |a, &b| a.max(b));
    let scale_u8 = 255.0 / (max_w_raw - min_w);
    let w_u8: Vec<u8> = w
        .iter()
        .map(|&v| ((v - min_w) * scale_u8).round() as u8)
        .collect();
    let err_u8: f32 = w
        .iter()
        .zip(w_u8.iter())
        .map(|(&v, &q)| (v - (q as f32 / scale_u8 + min_w)).powi(2))
        .sum::<f32>()
        .sqrt();

    println!("\n=== quantize Type Comparison (d={}) ===", d);
    println!(
        "i8 (Symmetric) RMSE Error: {:.8}",
        err_i8 / (d as f32).sqrt()
    );
    println!(
        "u8 (Min-Max)   RMSE Error: {:.8}",
        err_u8 / (d as f32).sqrt()
    );
    println!(
        "Winning Type: {}",
        if err_i8 < err_u8 { "i8" } else { "u8" }
    );

    // 2. IntMM のベンチマーク (i8 を採用)
    let mut mp = IntMM::new(d, d_out);
    mp.train(&xs, &vec![w.clone()]);

    let mut exacts = Vec::with_capacity(test_n);
    let mut approxs_f32_f32 = Vec::with_capacity(test_n);
    let mut approxs_i8_f32 = Vec::with_capacity(test_n);

    // 量子化した入力を事前に用意
    let xs_i8: Vec<Vec<i8>> = xs
        .iter()
        .map(|x| {
            x.iter()
                .map(|&v| (v * mp.input_scale).round().clamp(-128.0, 127.0) as i8)
                .collect()
        })
        .collect();

    // 速度計測
    let start_exact = Instant::now();
    for i in 0..test_n {
        let x = &xs[i];
        let exact: f32 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
        exacts.push(exact);
    }
    let dur_exact = start_exact.elapsed();

    let start_f32_f32 = Instant::now();
    for i in 0..test_n {
        approxs_f32_f32.push(mp.product_f32_f32(&xs[i])[0]);
    }
    let dur_f32_f32 = start_f32_f32.elapsed();

    let start_i8_f32 = Instant::now();
    for i in 0..test_n {
        approxs_i8_f32.push(mp.product_i8_f32(&xs_i8[i], mp.input_scale)[0]);
    }
    let dur_i8_f32 = start_i8_f32.elapsed();

    println!("\n=== IntMM Performance (n={}, d_out=1) ===", test_n);
    println!(
        "Exact Dot (f32):      {:.2} ns",
        dur_exact.as_nanos() as f64 / test_n as f64
    );
    println!(
        "IntMM f32 -> f32:     {:.2} ns",
        dur_f32_f32.as_nanos() as f64 / test_n as f64
    );
    println!(
        "IntMM i8  -> f32:     {:.2} ns",
        dur_i8_f32.as_nanos() as f64 / test_n as f64
    );

    evaluate_metrics(
        "IntMM f32->f32 Accuracy",
        &exacts,
        &approxs_f32_f32,
        dur_f32_f32.as_nanos(),
    );
    evaluate_metrics(
        "IntMM i8 ->f32 Accuracy",
        &exacts,
        &approxs_i8_f32,
        dur_i8_f32.as_nanos(),
    );
}

#[test]
fn test_int_mm_matrix() {
    use crate::quantize::int_quantize::IntMM;
    use crate::utills::rand::get_random_normal;
    use std::time::Instant;

    let d_in = 256;
    let d_out = 128;
    let n = 1000;
    let test_n = 100;

    let ws = (0..d_out)
        .map(|_| get_random_normal(d_in, 0.0, 1.0))
        .collect::<Vec<_>>();
    let xs = (0..n)
        .map(|_| get_random_normal(d_in, 0.0, 1.0))
        .collect::<Vec<_>>();

    let mut mp = IntMM::new(d_in, d_out);
    mp.train(&xs, &ws);

    // 速度計測: 行列積としてのトータルレイテンシ
    let mut dummy_exact = 0.0f32;
    let start_exact = Instant::now();
    for i in 0..test_n {
        let x = &xs[i];
        for w in &ws {
            let res: f32 = x.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            dummy_exact += res;
        }
    }
    let dur_exact = start_exact.elapsed();

    let mut dummy_intmm_f32 = 0.0f32;
    let start_intmm_f32 = Instant::now();
    for i in 0..test_n {
        let res = mp.product_f32_f32(&xs[i]);
        dummy_intmm_f32 += res.iter().sum::<f32>();
    }
    let dur_intmm_f32 = start_intmm_f32.elapsed();

    // 量子化済み入力での計測
    let xs_i8: Vec<Vec<i8>> = xs
        .iter()
        .map(|x| {
            x.iter()
                .map(|&v| (v * mp.input_scale).round().clamp(-128.0, 127.0) as i8)
                .collect()
        })
        .collect();

    let mut dummy_intmm_i8 = 0.0f32;
    let start_intmm_i8 = Instant::now();
    for i in 0..test_n {
        let res = mp.product_i8_f32(&xs_i8[i], mp.input_scale);
        dummy_intmm_i8 += res.iter().sum::<f32>();
    }
    let dur_intmm_i8 = start_intmm_i8.elapsed();

    println!(
        "\n=== IntMM Matrix Performance (n={}, d_in={}, d_out={}) ===",
        test_n, d_in, d_out
    );
    println!(
        "Exact Matrix Mul Avg:   {:.2} ns",
        dur_exact.as_nanos() as f64 / test_n as f64
    );
    println!(
        "IntMM f32 -> f32 Avg:   {:.2} ns",
        dur_intmm_f32.as_nanos() as f64 / test_n as f64
    );
    println!(
        "IntMM i8  -> f32 Avg:   {:.2} ns",
        dur_intmm_i8.as_nanos() as f64 / test_n as f64
    );
    println!(
        "Exact Speedup (f32):    {:.2}x",
        (dur_exact.as_nanos() as f64) / (dur_intmm_f32.as_nanos() as f64)
    );
    println!(
        "Exact Speedup (i8):     {:.2}x",
        (dur_exact.as_nanos() as f64) / (dur_intmm_i8.as_nanos() as f64)
    );
    println!(
        "(Check sums: exact={}, f32={}, i8={})",
        dummy_exact, dummy_intmm_f32, dummy_intmm_i8
    );
}

fn assert_tensor(a: Tensor, b: Tensor, message: String) {
    assert_eq!(
        a.shape.len(),
        b.shape.len(),
        "{}[assert_tensor] shape dimension num is not same, a.shape.len()={}, b.shape.len()={}",
        message,
        a.shape.len(),
        b.shape.len()
    );
    for (i, (a_i, b_i)) in a.shape.iter().zip(b.shape.iter()).enumerate() {
        assert_eq!(
            a_i, b_i,
            "{}[assert_tensor] shape size is not same a.shape[{i}]={a_i}, b.shape[{i}]={b_i}",
            message
        );
    }

    let a_f32 = a.as_f32_slice();
    let b_f32 = b.as_f32_slice();
    for (i, (a_i, b_i)) in a_f32.iter().zip(b_f32.iter()).enumerate() {
        assert!(
            (a_i - b_i).abs() < 1e-5,
            "{}[assert_tensor] data is not same a.data[{i}]={a_i}, b.data[{i}]={b_i}",
            message
        );
    }
}
