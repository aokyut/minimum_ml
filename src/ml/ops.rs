use super::funcs::SingleShoot;
use super::Node;
use super::Tensor;

pub struct Add {
    ignore_grad: bool,
}

impl Default for Add {
    fn default() -> Self {
        Self::new()
    }
}

impl Add {
    pub fn new() -> Self {
        Add { ignore_grad: false }
    }
}

impl Node for Add {
    fn backward(&mut self, grad: &Tensor, inputs: Vec<&Tensor>, _: &Tensor) -> Vec<Tensor> {
        let mut outs = Vec::new();

        for _ in 0..inputs.len() {
            outs.push(grad.clone());
        }

        outs
    }
    fn call(&self, inputs: Vec<super::Tensor>) -> super::Tensor {
        let mut out_vec = inputs[0].as_f32_slice().into_owned();
        let size = out_vec.len();
        for i in 1..inputs.len() {
            let in_data = inputs[i].as_f32_slice();
            assert_eq!(size, in_data.len());
            for j in 0..size {
                out_vec[j] += in_data[j];
            }
        }
        super::Tensor::new(out_vec, inputs[0].shape.clone())
    }
    fn no_grad(&self) -> bool {
        self.ignore_grad
    }
}

pub struct MulConst {
    c: f32,
}

impl MulConst {
    pub fn new(c: f32) -> Self {
        MulConst { c }
    }
}

impl SingleShoot for MulConst {
    fn single_forward(&self, x: f32) -> f32 {
        self.c * x
    }
    fn single_backward(&self, _: f32, _: f32) -> f32 {
        self.c
    }
}
