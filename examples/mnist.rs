use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::Hyperparams;
use smartmetal::net::load_mnist::load_data;

use std::path::Path;

fn main() {
    let data =
        load_data("/home/eu/programming/dl-framework/data/train").expect("Could not load dataset.");
    println!("MNIST dataset has been loaded");

    let arch = vec![28 * 28, 512, 512, 10];

    let params = Hyperparams::new(5, 1e-2);
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 3];
    act_funcs.push(ActivationFunction::Softmax);

    let mut net = Net::new(arch, act_funcs, loss_func);
    println!("Neural network has been initialized\n");

    println!("Training has started...");
    net.train(&data.images, &data.classes, &params);
    println!("Training ended.");

    let path = Path::new("/home/eu/programming/dl-framework/models/mnist.json");
    println!("Saving model...");
    net.save_to(path)
        .expect(&format!("Failed to save to {:?}", path));
    println!("Done.");
}
