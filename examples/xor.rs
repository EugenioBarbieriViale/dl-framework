use crate::net::Net;
use net::functions::{ActivationFunction, LossFunction};

use nalgebra::DMatrix;
use std::path::Path;

mod net;

pub struct Hyperparams {
    epochs: usize,
    learning_rate: f64,
}

impl Hyperparams {
    pub fn new() -> Self {
        Hyperparams {
            epochs: 100,
            learning_rate: 1e-3,
        }
    }
}

fn main() {
    let data = DMatrix::from_row_slice(8, 1, &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    let label = DMatrix::from_row_slice(4, 1, &[0.0, 1.0, 1.0, 0.0]);

    let arch = vec![8, 6, 4];

    let hypp = Hyperparams::new();
    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; 3];

    let mut net = Net::new(arch, act_funcs, loss_func);

    net.train(&data, &label, &hypp);

    let path = Path::new("/home/eu/programming/dl-framework/models/xor.json");

    println!("Saving model...");
    net.save_to(path).unwrap();
    println!("Done");

    println!("Loading model...");
    let p = net.load_from(path).unwrap();
    println!("Done");

    net.params = p;

    let out = net.predict(&data);
    println!("{:?}", data);
    println!("{:?}", label);
    println!("------------");
    println!("{:?}", out);
}
