use std::collections::HashMap;

use super::{Optimizer, Tensor};

pub struct SGD {
    alpha: f32,
}

impl SGD {
    pub fn new(alpha: f32) -> Self {
        return Self { alpha: alpha };
    }
}

impl Optimizer for SGD {
    fn optimize(&mut self, _: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        let mut ans_vec = Vec::new();

        for grad in grads {
            let mut ans = Tensor::zeros_like(grad);
            let grad_f32 = grad.as_f32_slice();
            let ans_f32 = ans.f32_data_mut();
            for i in 0..ans_f32.len() {
                ans_f32[i] = -grad_f32[i] * self.alpha;
            }
            ans_vec.push(ans);
        }

        return ans_vec;
    }
}

pub struct MomentumSGD {
    alpha: f32,
    beta: f32,
    logs: HashMap<usize, Vec<Tensor>>,
}

impl MomentumSGD {
    // TODO
    pub fn new(alpha: f32, beta: f32) -> Self {
        return MomentumSGD {
            alpha: alpha,
            beta: beta,
            logs: HashMap::new(),
        };
    }
}

impl Optimizer for MomentumSGD {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        let log = self.logs.get(&tar_id);
        if let Some(vn) = log {
            let mut w = Vec::new();
            let mut vn_ = Vec::new();
            for i in 0..grads.len() {
                let vni = &vn[i];
                let grad = grads[i];
                let mut vni_ = Tensor::zeros_like(grad);
                let mut wi = Tensor::zeros_like(grad);
                
                let vni_f32 = vni.as_f32_slice();
                let grad_f32 = grad.as_f32_slice();
                let vni_new_f32 = vni_.f32_data_mut();
                let wi_f32 = wi.f32_data_mut();

                for j in 0..grad_f32.len() {
                    vni_new_f32[j] = self.beta * vni_f32[j] + (1.0 - self.beta) * grad_f32[j];
                    wi_f32[j] = -self.alpha * vni_new_f32[j];
                }

                w.push(wi);
                vn_.push(vni_);
            }
            self.logs.entry(tar_id).and_modify(|e| *e = vn_);
            return w;
        } else {
            let mut w = Vec::new();
            let mut vn: Vec<Tensor> = Vec::new();
            for i in 0..grads.len() {
                let grad = grads[i];
                let mut vni = Tensor::zeros_like(grad);
                let mut wi = Tensor::zeros_like(grad);
                
                let grad_f32 = grad.as_f32_slice();
                let vni_f32 = vni.f32_data_mut();
                let wi_f32 = wi.f32_data_mut();

                for j in 0..grad_f32.len() {
                    vni_f32[j] = grad_f32[j];
                    wi_f32[j] = -self.alpha * vni_f32[j];
                }
                vn.push(vni);
                w.push(wi);
            }
            self.logs.insert(tar_id, vn);
            return w;
        }
    }
}

pub struct RMSProp {
    alpha: f32,
    beta: f32,
    epsilon: f32,
    v: HashMap<usize, Vec<Tensor>>,
}

impl RMSProp {
    pub fn new(alpha: f32, beta: f32) -> Self {
        return RMSProp {
            alpha,
            beta,
            epsilon: 1e-8,
            v: HashMap::new(),
        };
    }

    pub fn with_epsilon(alpha: f32, beta: f32, epsilon: f32) -> Self {
        return RMSProp {
            alpha,
            beta,
            epsilon,
            v: HashMap::new(),
        };
    }
}

impl Optimizer for RMSProp {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        if !self.v.contains_key(&tar_id) {
            let mut initial_v = Vec::new();
            for grad in &grads {
                initial_v.push(Tensor::zeros_like(grad));
            }
            self.v.insert(tar_id, initial_v);
        }

        let vs = self.v.get_mut(&tar_id).unwrap();
        let mut updates = Vec::new();

        for i in 0..grads.len() {
            let grad = grads[i];
            let vi = &mut vs[i];
            let mut update = Tensor::zeros_like(grad);

            let grad_f32 = grad.as_f32_slice();
            let vi_f32 = vi.f32_data_mut();
            let update_f32 = update.f32_data_mut();

            for j in 0..grad_f32.len() {
                vi_f32[j] = self.beta * vi_f32[j] + (1.0 - self.beta) * grad_f32[j] * grad_f32[j];
                update_f32[j] = -self.alpha * grad_f32[j] / (vi_f32[j].sqrt() + self.epsilon);
            }
            updates.push(update);
        }

        return updates;
    }
}

pub struct Adam {
    alpha: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    m: HashMap<usize, Vec<Tensor>>,
    v: HashMap<usize, Vec<Tensor>>,
    t: HashMap<usize, u32>,
}

impl Adam {
    pub fn new(alpha: f32, beta1: f32, beta2: f32) -> Self {
        return Adam {
            alpha,
            beta1,
            beta2,
            epsilon: 1e-8,
            m: HashMap::new(),
            v: HashMap::new(),
            t: HashMap::new(),
        };
    }

    pub fn with_epsilon(alpha: f32, beta1: f32, beta2: f32, epsilon: f32) -> Self {
        return Adam {
            alpha,
            beta1,
            beta2,
            epsilon,
            m: HashMap::new(),
            v: HashMap::new(),
            t: HashMap::new(),
        };
    }
}

impl Optimizer for Adam {
    fn optimize(&mut self, tar_id: usize, grads: Vec<&Tensor>) -> Vec<Tensor> {
        if !self.m.contains_key(&tar_id) {
            let mut initial_m = Vec::new();
            let mut initial_v = Vec::new();
            for grad in &grads {
                initial_m.push(Tensor::zeros_like(grad));
                initial_v.push(Tensor::zeros_like(grad));
            }
            self.m.insert(tar_id, initial_m);
            self.v.insert(tar_id, initial_v);
            self.t.insert(tar_id, 0);
        }

        let time = self.t.get_mut(&tar_id).unwrap();
        *time += 1;
        let t = *time as f32;

        let ms = self.m.get_mut(&tar_id).unwrap();
        let vs = self.v.get_mut(&tar_id).unwrap();
        let mut updates = Vec::new();

        let lr_t = self.alpha * (1.0 - self.beta2.powf(t)).sqrt() / (1.0 - self.beta1.powf(t));

        for i in 0..grads.len() {
            let grad = grads[i];
            let mi = &mut ms[i];
            let vi = &mut vs[i];
            let mut update = Tensor::zeros_like(grad);

            let grad_f32 = grad.as_f32_slice();
            let mi_f32 = mi.f32_data_mut();
            let vi_f32 = vi.f32_data_mut();
            let update_f32 = update.f32_data_mut();

            for j in 0..grad_f32.len() {
                mi_f32[j] = self.beta1 * mi_f32[j] + (1.0 - self.beta1) * grad_f32[j];
                vi_f32[j] = self.beta2 * vi_f32[j] + (1.0 - self.beta2) * grad_f32[j] * grad_f32[j];
                
                // Efficiently combine bias correction into the learning rate
                update_f32[j] = -lr_t * mi_f32[j] / (vi_f32[j].sqrt() + self.epsilon);
            }
            updates.push(update);
        }

        return updates;
    }
}




















































