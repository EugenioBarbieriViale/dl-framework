// mod load_mnist;
// mod net;
//
// // use nalgebra::DMatrix;
//
// use crate::net::Net;
// use net::functions::ActivationFunction;
// use net::functions::LossFunction;
//
// use load_mnist::load_data;
//
// pub struct Hyperparams {
//     epochs: usize,
//     learning_rate: f64,
// }
//
// impl Hyperparams {
//     pub fn new() -> Self {
//         Hyperparams {
//             epochs: 100,
//             learning_rate: 1e-2,
//         }
//     }
// }
//
// fn main() {
//     let data = load_data("/home/eu/programming/dl-framework/data/train").unwrap();
//     println!("MNIST dataset has been loaded");
//
//     let data_size = data.len();
//
//     let arch = vec![28 * 28, 1024, 512, 10];
//
//     let params = Hyperparams::new();
//     let loss_func = LossFunction::CrossEntropy;
//     let mut act_funcs = vec![ActivationFunction::ReLU; 3];
//     act_funcs.push(ActivationFunction::Softmax);
//
//     let mut net = Net::new(arch, act_funcs, loss_func);
//     println!("Neural network has been initialized\n");
//
//     for e in 0..params.epochs {
//         for i in 0..data_size {
//             net.train(&data[i].image, &data[i].class, &params);
//
//             if i % 100 == 0 {
//                 println!("{} -> [{}%]", i, i / data_size * 100);
//             }
//         }
//
//         println!("\nEpoch: {} cost: {}", e, net.cost);
//         println!("-------------------------------");
//     }
// }

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
            epochs: 10000,
            learning_rate: 1e-2,
        }
    }
}

fn main() {
    let data = DMatrix::from_row_slice(2, 4, &[0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    let label = DMatrix::from_row_slice(1, 4, &[0.0, 1.0, 1.0, 0.0]);

    let arch = vec![2, 4, 1];

    let params = Hyperparams::new();
    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; 3];
    // let act_funcs = vec![ActivationFunction::Sigmoid; 3];

    let mut net = Net::new(arch, act_funcs, loss_func);
    let (mut nabla_w, mut nabla_b) = net.init_gradients();

    for e in 0..params.epochs {
        for i in 0..4 {
            let x = get_col(i, &data);
            let y = get_col(i, &label);

            net.train(&x, &y, &mut nabla_w, &mut nabla_b, &params);
        }

        if e % 100 == 0 {
            println!("Epoch: {} cost: {}", e, net.cost);
        }
    }

    for i in 0..4 {
        let x = get_col(i, &data);
        let y = get_col(i, &label);

        let out = net.predict(&x);

        println!(
            "{}, {} -> {}: {}",
            x[(0, 0)],
            x[(1, 0)],
            y[(0, 0)],
            out[(0, 0)]
        );
    }

    println!("Saving model...");
    net.save_to(Path::new("../model.json")).unwrap();
    println!("Done");
}

fn get_col(i: usize, m: &DMatrix<f64>) -> DMatrix<f64> {
    DMatrix::from_vec(m.nrows(), 1, m.column(i).iter().cloned().collect())
}
