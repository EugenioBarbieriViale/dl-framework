use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::Hyperparams;

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

    let hypp = Hyperparams::new(1000, 1, 1e-2);
    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; 2];

    let mut net = Net::new(arch, act_funcs, loss_func);

    net.seq_train(&data, &label, &hypp);

    // let path = Path::new("/home/eu/programming/dl-framework/models/xor.json");
    //
    // println!("Saving model...");
    // net.save_to(path).unwrap();
    // println!("Done");
    //
    // println!("Loading model...");
    // net.load_from(path).unwrap();
    // println!("Done");

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
