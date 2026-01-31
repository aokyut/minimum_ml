pub fn half_imcomplete_beta_func(gamma: f64, delta: f64) -> f64 {
    let mut c0 = 1.0;
    // (g + d + 1)!/(g!d!) * (1/2)^(g + d + 1)

    let mut stack = vec![(gamma, delta)];

    loop {
        // print!("flag");
        let (g, d) = stack.pop().unwrap();

        if g == d && d == 0.0 {
            break;
        }
        if g < d {
            c0 *= 0.5 * (g + d + 1.0) / d;
            stack.push((g, d - 1.0));
        } else {
            c0 *= 0.5 * (g + d + 1.0) / g;
            stack.push((d, g - 1.0));
        }
    }
    c0 *= 0.5;

    let mut a = c0 / (gamma + delta + 1.0);

    for i in 1..=(delta as usize) {
        // println!("{i}");
        a = (a * (i as f64) + c0) / (gamma + delta + 1.0 - i as f64);
    }

    a
}

pub mod rand {
    use rand::prelude::*;
    pub fn get_random_usize() -> usize {
        let mut rng = rand::thread_rng();
        rng.random::<u64>() as usize
    }
    pub fn get_random_usizes(size: usize) -> Vec<usize> {
        let mut v = vec![0; size];
        let mut rng = rand::thread_rng();
        for i in 0..size {
            v[i] = rng.random::<u64>() as usize;
        }
        v
    }
    pub fn get_random_normal(size: usize, mu: f32, sigma: f32) -> Vec<f32> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut ans = Vec::with_capacity(size);
        
        // Box-Muller変換で正規分布を生成
        let mut i = 0;
        while i < size {
            let u1: f32 = rng.r#gen();
            let u2: f32 = rng.r#gen();
            
            // 2つの独立な正規分布乱数を生成
            let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos();
            let z1 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).sin();
            
            ans.push(mu + sigma * z0);
            i += 1;
            
            if i < size {
                ans.push(mu + sigma * z1);
                i += 1;
            }
        }
        
        ans
    }

    pub struct RandUsizeGenerator {
        rng: ThreadRng,
    }

    impl Iterator for RandUsizeGenerator {
        type Item = usize;
        fn next(&mut self) -> Option<usize> {
            Some(self.rng.random::<u64>() as usize)
        }
    }
}
