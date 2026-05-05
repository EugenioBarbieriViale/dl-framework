use functions::*;
use hyperparams::*;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

use crate::net::hyperparams::Initialization;

pub mod checkpoint;
pub mod functions;
pub mod hyperparams;
pub mod load_mnist;
pub mod train;

#[derive(Serialize, Deserialize, Debug)]
pub struct NetParams {
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
}

impl NetParams {
    pub fn new(arch: &Vec<usize>, layers: usize, init: &Initialization) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for i in 0..layers {
            let c = arch[i];
            let r = arch[i + 1];

            let weight_matrix = new(init, (r, c));
            let bias_matrix = new(init, (r, 1));

            weights.push(weight_matrix);
            biases.push(bias_matrix);
        }

        NetParams { weights, biases }
    }
}

#[derive(Debug)]
pub struct Net {
    layers: usize,

    act_functions: Vec<ActivationFunction>,
    loss_function: LossFunction,

    pub params: NetParams,
    zs: Vec<DMatrix<f64>>,
    activations: Vec<DMatrix<f64>>,
}

impl Net {
    pub fn new(
        arch: Vec<usize>,
        act_functions: Vec<ActivationFunction>,
        loss_function: LossFunction,
        init: &Initialization,
    ) -> Self {
        let layers = arch.len() - 1;

        if layers != act_functions.len() {
            panic!(
                "Network and activation functions sizes mismatch ({}, {})",
                layers,
                act_functions.len()
            );
        }

        let params = NetParams::new(&arch, layers, init);

        let mut zs = Vec::with_capacity(layers);
        let mut activations = Vec::with_capacity(layers + 1);

        for i in 0..=layers {
            let size = arch[i];
            activations.push(DMatrix::<f64>::zeros(size, 1));
            if i < layers {
                zs.push(DMatrix::<f64>::zeros(arch[i + 1], 1));
            }
        }

        Net {
            layers,
            act_functions,
            loss_function,
            params,
            zs,
            activations,
        }
    }

    fn init_gradients(&self) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut nabla_w: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);
        let mut nabla_b: Vec<DMatrix<f64>> = Vec::with_capacity(self.layers);

        for i in 0..self.layers {
            nabla_w.push(DMatrix::zeros(
                self.params.weights[i].nrows(),
                self.params.weights[i].ncols(),
            ));
            nabla_b.push(DMatrix::zeros(
                self.params.biases[i].nrows(),
                self.params.biases[i].ncols(),
            ));
        }

        (nabla_w, nabla_b)
    }
}
