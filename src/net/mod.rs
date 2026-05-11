use crate::net::init::*;
use functions::*;
// use hyperparams::*;

use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};

mod buffer;
pub mod checkpoint;
pub mod functions;
pub mod hyperparams;
pub mod init;
pub mod load_mnist;
pub mod train;

#[derive(Serialize, Deserialize, Debug)]
pub struct NetParams {
    weights: Vec<DMatrix<f64>>,
    biases: Vec<DMatrix<f64>>,
}

impl NetParams {
    pub fn new(arch: &Vec<usize>, layers: usize, init: &Initialization) -> Self {
        let mut weights = Vec::with_capacity(layers);
        let mut biases = Vec::with_capacity(layers);

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
    arch: Vec<usize>,

    act_functions: Vec<ActivationFunction>,
    loss_function: LossFunction,

    pub params: NetParams,
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

        Net {
            layers,
            arch,
            act_functions,
            loss_function,
            params,
        }
    }
}
