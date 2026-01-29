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

    return a;
}

pub mod rand {
    use rand::prelude::*;
    pub fn get_random_usize() -> usize {
        let mut rng = rand::rng();
        return rng.random::<u64>() as usize;
    }
    pub fn get_random_usizes(size: usize) -> Vec<usize> {
        let mut v = vec![0; size];
        let mut rng = rand::rng();
        for i in 0..size {
            v[i] = rng.random::<u64>() as usize;
        }
        return v;
    }
    pub fn get_random_normal(size: usize, mu: f32, sigma: f32) -> Vec<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rng();
        let normal = Normal::new(mu, sigma).unwrap();
        let mut ans = Vec::new();
        for _ in 0..size {
            ans.push(normal.sample(&mut rng));
        }
        return ans;
    }

    pub struct RandUsizeGenerator {
        rng: ThreadRng,
    }

    impl Iterator for RandUsizeGenerator {
        type Item = usize;
        fn next(&mut self) -> Option<usize> {
            return Some(self.rng.random::<u64>() as usize);
        }
    }
}
