use crate::net::NetParams;
use crate::net::init::Initialization;

use nalgebra::DMatrix;

pub struct Buffer {
    pub zs: Vec<DMatrix<f64>>,
    pub activations: Vec<DMatrix<f64>>,
    pub nabla_w: Vec<DMatrix<f64>>,
    pub nabla_b: Vec<DMatrix<f64>>,
}

impl Buffer {
    pub fn alloc(arch: &Vec<usize>, layers: usize) -> Self {
        let mut zs = Vec::with_capacity(layers);
        let mut activations = Vec::with_capacity(layers + 1);

        for i in 0..=layers {
            activations.push(DMatrix::<f64>::zeros(arch[i], 1));
            if i < layers {
                zs.push(DMatrix::<f64>::zeros(arch[i + 1], 1));
            }
        }

        let nabla = NetParams::new(arch, layers, &Initialization::Zero);

        Self {
            zs,
            activations,
            nabla_w: nabla.weights,
            nabla_b: nabla.biases,
        }
    }

    pub fn alloc_grad(arch: &Vec<usize>, layers: usize) -> NetParams {
        NetParams::new(arch, layers, &Initialization::Zero)
    }

    pub fn zero(&mut self, layers: usize) {
        for i in 0..=layers {
            self.activations[i].fill(0.0);
            if i < layers {
                self.zs[i].fill(0.0);
            }
        }
    }

    pub fn zero_grad(&mut self, layers: usize) {
        for i in 0..layers {
            self.nabla_w[i].fill(0.0);
            self.nabla_b[i].fill(0.0);
        }
    }
}
