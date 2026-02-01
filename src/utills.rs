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
    use std::cell::RefCell;
    use std::sync::Once;

    // Xorshift64 implementation
    pub struct XorShift64 {
        state: u64,
    }

    impl XorShift64 {
        pub fn new(seed: u64) -> Self {
            Self { state: if seed == 0 { 88172645463325252 } else { seed } }
        }

        pub fn from_entropy() -> Self {
            let mut seed_buf = [0u8; 8];
            getrandom::getrandom(&mut seed_buf).expect("failed to get random seed");
            let seed = u64::from_le_bytes(seed_buf);
            Self::new(seed)
        }

        pub fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }

        pub fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }

        pub fn gen_f32(&mut self) -> f32 {
            // Generate float in [0, 1)
            // 24 bits of randomness
            let v = self.next_u32() >> 8;
            (v as f32) * (1.0 / 16777216.0)
        }
        
        pub fn gen_range(&mut self, start: usize, end: usize) -> usize {
            if start >= end {
                return start;
            }
            let range = (end - start) as u64;
            // Simple modulo, slight bias possible but fine for this purpose
            start + (self.next_u64() % range) as usize
        }
        
        // Fisher-Yates shuffle
        pub fn shuffle<T>(&mut self, slice: &mut [T]) {
            for i in (1..slice.len()).rev() {
                let j = self.gen_range(0, i + 1);
                slice.swap(i, j);
            }
        }
    }

    thread_local! {
        static RNG: RefCell<XorShift64> = RefCell::new(XorShift64::from_entropy());
    }

    // Public API to mimic rand usage
    pub fn rng() -> impl RngCore {
        struct ThreadRngProxy;
        impl RngCore for ThreadRngProxy {
            fn gen_range(&mut self, start: usize, end: usize) -> usize {
                RNG.with(|rng| rng.borrow_mut().gen_range(start, end))
            }
            fn gen_f32(&mut self) -> f32 {
                RNG.with(|rng| rng.borrow_mut().gen_f32())
            }
            fn shuffle<T>(&mut self, slice: &mut [T]) {
                RNG.with(|rng| rng.borrow_mut().shuffle(slice))
            }
        }
        ThreadRngProxy
    }

    pub trait RngCore {
        fn gen_range(&mut self, start: usize, end: usize) -> usize;
        fn gen_f32(&mut self) -> f32;
        fn shuffle<T>(&mut self, slice: &mut [T]);
    }

    pub fn get_random_usize() -> usize {
        RNG.with(|rng| rng.borrow_mut().next_u64() as usize)
    }

    pub fn get_random_usizes(size: usize) -> Vec<usize> {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            let mut v = vec![0; size];
            for i in 0..size {
                v[i] = r.next_u64() as usize;
            }
            v
        })
    }
    
    pub fn get_random_normal(size: usize, mu: f32, sigma: f32) -> Vec<f32> {
        RNG.with(|rng| {
            let mut r = rng.borrow_mut();
            let mut ans = Vec::with_capacity(size);
            
            // Box-Muller transform
            let mut i = 0;
            while i < size {
                let u1: f32 = r.gen_f32();
                let u2: f32 = r.gen_f32();
                
                // Avoid log(0)
                let u1 = if u1 < 1e-7 { 1e-7 } else { u1 };
                
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
        })
    }
}
