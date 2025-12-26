mod net;
mod load_mnist;

// use nalgebra::DMatrix;

use crate::net::Net;
use net::functions::LossFunction;
use net::functions::ActivationFunction;

use load_mnist::load_data;



pub struct Hyperparams {
    epochs: usize,
    learning_rate: f64,
}

impl Hyperparams {
    pub fn new() -> Self {
        Hyperparams {
            epochs: 100,
            learning_rate: 1e-2,
        }
    }
}


fn main() {
    let data = load_data("/home/eu/programming/dl-framework/data/train").unwrap();
    println!("MNIST dataset has been loaded");

    let data_size = data.len();

    let arch = vec![28*28, 1024, 512, 10];

    let params = Hyperparams::new();
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 3];
    act_funcs.push(ActivationFunction::Softmax);

    let mut net = Net::new(arch, act_funcs, loss_func);
    println!("Neural network has been initialized\n");

    for e in 0..params.epochs {
        for i in 0..data_size {
            net.train(&data[i].image, &data[i].class, &params);

            if i % 100 == 0 {
                println!("{} -> [{}%]", i, i / data_size * 100);
            }
        }

        println!("\nEpoch: {} cost: {}", e, net.cost);
        println!("-------------------------------");
    }
}
