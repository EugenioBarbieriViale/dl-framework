use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::{Hyperparams, Initialization};

use nalgebra::DMatrix;
use std::path::Path;

fn main() {
    let data: Vec<DMatrix<f64>> = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        .chunks(2)
        .map(|s| DMatrix::from_row_slice(2, 1, s))
        .collect();

    let label: Vec<DMatrix<f64>> = [0.0, 1.0, 1.0, 0.0]
        .chunks(1)
        .map(|s| DMatrix::from_row_slice(1, 1, s))
        .collect();

    let arch = vec![2, 2, 1];

    let hypp = Hyperparams::new(10000, 1, 0.7);
    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::Sigmoid; 2];
    let init = Initialization::Random;

    let mut net = Net::new(arch, act_funcs, loss_func, &init);

    net.seq_train(&data, &label, &hypp);

    let path = Path::new("/home/eu/programming/smart-metal/models/xor.json");

    net.save_to(path).unwrap();
    net.load_from(path).unwrap();

    for (x, y) in data.into_iter().zip(label.into_iter()) {
        let out = net.predict_raw(&x);
        println!(
            "{} {}: {} => {}",
            x[(0, 0)],
            x[(1, 0)],
            y[(0, 0)],
            out[(0, 0)]
        );
    }
}
