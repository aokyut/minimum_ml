#[cfg(feature = "serialization")]
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serialization", derive(Serialize, Deserialize))]
pub struct IntMM {
    pub d_in: usize,
    pub d_out: usize,
    pub quantized_w: Vec<i8>, // Flattened [d_out, d_in]
    pub w_scale: f32,
    pub input_scale: f32,
    pub output_scale: f32,
}

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
impl IntMM {
    pub fn new(d_in: usize, d_out: usize) -> Self {
        Self {
            d_in,
            d_out,
            quantized_w: vec![],
            w_scale: 1.0,
            input_scale: 1.0,
            output_scale: 1.0,
        }
    }

    pub fn train(&mut self, xs: &Vec<Vec<f32>>, w: &Vec<Vec<f32>>) {
        // 全ての重みの中での最大値を見つける
        let mut max_w = 0.0f32;
        let mut max_y = 0.0f32;

        for row in w {
            let mut sum_horizon = 0.0f32;
            for &v in row {
                max_w = max_w.max(v.abs());
                sum_horizon += v.abs();
            }
            max_y = max_y.max(sum_horizon);
        }
        self.w_scale = if max_w > 0.0 { 127.0 / max_w } else { 1.0 };

        self.quantized_w = Vec::with_capacity(self.d_out * self.d_in);
        for row in w {
            for &v in row {
                self.quantized_w.push((v * self.w_scale).round() as i8);
            }
        }

        // 入力のスケールを推定
        let mut max_x = 0.0f32;
        let mut observed_max_y = 0.0f32;

        if !xs.is_empty() {
            for x in xs.iter().take(200) {
                let mut local_max_x = 0.0f32;
                for &v in x {
                    local_max_x = local_max_x.max(v.abs());
                }
                max_x = max_x.max(local_max_x);

                // 実際の出力を計算して、出力レンジを測定（キャリブレーション）
                for k in 0..self.d_out {
                    let mut sum = 0.0f32;
                    let row_w = &w[k];
                    for i in 0..self.d_in {
                        sum += x[i] * row_w[i];
                    }
                    observed_max_y = observed_max_y.max(sum.abs());
                }
            }
        } else {
            // データがない場合のデフォルト（小さめのサンプル数で見積もり）
            max_x = 1.0; // 仮定
        }

        self.input_scale = if max_x > 0.0 { 127.0 / max_x } else { 1.0 };

        if observed_max_y > 0.0 {
            // キャリブレーションデータに基づく正確なスケール
            self.output_scale = 127.0 / observed_max_y;
        } else {
            // データがない場合の論理的な最大値を使用
            // ただし、全ての要素が同時に最大値になることは稀なので、
            // 統計的な見積もり（sqrt(d_in)）を導入して少し強気に振る
            let statistical_max_y = max_x * max_w * (self.d_in as f32).sqrt();
            self.output_scale = if statistical_max_y > 0.0 {
                127.0 / statistical_max_y
            } else {
                1.0
            };
        }
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn dot_product_i8_avx2(a: &[i8], b: &[i8]) -> i32 {
        let n = a.len();
        let mut chunk_idx = 0;
        let mut sum_vec = unsafe { _mm256_setzero_si256() };

        while chunk_idx + 32 <= n {
            let (va, vb) = unsafe {
                (
                    _mm256_loadu_si256(a.as_ptr().add(chunk_idx) as *const __m256i),
                    _mm256_loadu_si256(b.as_ptr().add(chunk_idx) as *const __m256i),
                )
            };

            let (va_lo, vb_lo, va_hi, vb_hi) = unsafe {
                (
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 0)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 0)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(va, 1)),
                    _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vb, 1)),
                )
            };

            sum_vec = unsafe {
                let s = _mm256_add_epi32(sum_vec, _mm256_madd_epi16(va_lo, vb_lo));
                _mm256_add_epi32(s, _mm256_madd_epi16(va_hi, vb_hi))
            };

            chunk_idx += 32;
        }

        let res_arr: [i32; 8] = unsafe { std::mem::transmute(sum_vec) };
        let mut sum = res_arr.iter().sum::<i32>();

        while chunk_idx < n {
            sum += (a[chunk_idx] as i32) * (b[chunk_idx] as i32);
            chunk_idx += 1;
        }
        sum
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn quantize_f32_i8_avx2(input: &[f32], scale: f32, output: &mut [i8]) {
        let n = input.len();
        let scale_vec = unsafe { _mm256_set1_ps(scale) };
        let mut i = 0;

        while i + 8 <= n {
            let shifted = unsafe {
                let fin = _mm256_loadu_ps(input.as_ptr().add(i));
                let multiplied = _mm256_mul_ps(fin, scale_vec);
                _mm256_cvtps_epi32(_mm256_round_ps(
                    multiplied,
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC,
                ))
            };

            let res_arr: [i32; 8] = unsafe { std::mem::transmute(shifted) };
            for j in 0..8 {
                output[i + j] = res_arr[j].clamp(-128, 127) as i8;
            }
            i += 8;
        }
        while i < n {
            output[i] = (input[i] * scale).round().clamp(-128.0, 127.0) as i8;
            i += 1;
        }
    }

    pub fn product_f32_f32(&self, input: &[f32]) -> Vec<f32> {
        let mut x_q = vec![0i8; self.d_in];
        unsafe { Self::quantize_f32_i8_avx2(input, self.input_scale, &mut x_q) };

        let mut results = Vec::with_capacity(self.d_out);
        for k in 0..self.d_out {
            let row_w = &self.quantized_w[k * self.d_in..(k + 1) * self.d_in];
            let sum_q = unsafe { Self::dot_product_i8_avx2(&x_q, row_w) };
            results.push((sum_q as f32) / (self.input_scale * self.w_scale));
        }
        results
    }

    pub fn product_i8_f32(&self, input: &[i8], input_scale: f32) -> Vec<f32> {
        self.product_i8_f32_with_bias(input, input_scale, None)
    }

    pub fn product_i8_f32_with_bias(
        &self,
        input: &[i8],
        input_scale: f32,
        bias: Option<&[f32]>,
    ) -> Vec<f32> {
        let mut results = Vec::with_capacity(self.d_out);
        for k in 0..self.d_out {
            let row_w = &self.quantized_w[k * self.d_in..(k + 1) * self.d_in];
            let sum_q = unsafe { Self::dot_product_i8_avx2(input, row_w) };
            let mut val = (sum_q as f32) / (input_scale * self.w_scale);
            if let Some(b) = bias {
                val += b[k];
            }
            results.push(val);
        }
        results
    }

    pub fn product_f32_i8(&self, input: &[f32], bias: Option<&[f32]>) -> Vec<i8> {
        let mut x_q = vec![0i8; self.d_in];
        unsafe { Self::quantize_f32_i8_avx2(input, self.input_scale, &mut x_q) };
        self.product_i8_i8(&x_q, self.input_scale, bias)
    }

    pub fn product_i8_i8(&self, input: &[i8], input_scale: f32, bias: Option<&[f32]>) -> Vec<i8> {
        let mut results = Vec::with_capacity(self.d_out);
        let combined_scale = self.output_scale / (input_scale * self.w_scale);
        for k in 0..self.d_out {
            let row_w = &self.quantized_w[k * self.d_in..(k + 1) * self.d_in];
            let sum_q = unsafe { Self::dot_product_i8_avx2(input, row_w) };
            let mut out_f = (sum_q as f32) * combined_scale;
            if let Some(b) = bias {
                out_f += b[k] * self.output_scale;
            }
            results.push(out_f.round().clamp(-128.0, 127.0) as i8);
        }
        // println!("product_i8_i8: {:#?}", results);
        results
    }
}

/*
impl Productor for IntMM {
    fn product(&self, input: Tensor) -> Tensor {
        let n = input.shape[0]; // Batch size
        let mut out_data = Vec::with_capacity(n * self.d_out);
        let input_f32 = input.as_f32_slice();
        for i in 0..n {
            let row = &input_f32[i * self.d_in..(i + 1) * self.d_in];
            let result_vec = self.product_f32_f32(row);
            out_data.extend(result_vec);
        }
        Tensor::new(out_data, vec![n, self.d_out])
    }
}
*/
