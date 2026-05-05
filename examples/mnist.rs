use nalgebra::DMatrix;
use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::*;
use smartmetal::net::load_mnist::{load_data, one_hot_decode};

use std::path::Path;

// mnist: Accuracy of 96.39666666666666 %
// new_mnist: Accuracy of 99.05166666666668 %

fn main() {
    let data =
        load_data("/home/eu/programming/smart-metal/data/train").expect("Could not load dataset.");
    println!("MNIST dataset has been loaded");

    let arch = vec![28 * 28, 512, 512, 10];

    let params = Hyperparams::new(5, 32, 1e-2);
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 2];
    act_funcs.push(ActivationFunction::Softmax);
    let init = Initialization::Kaiming(Fan::In);

    let mut net = Net::new(arch, act_funcs, loss_func, &init);
    println!("Neural network has been initialized\n");

    // net.par_train(&data.images, &data.classes, &params);

    // let model_path = Path::new("/home/eu/programming/smart-metal/models/new_mnist.json");
    let model_path = Path::new("/home/eu/programming/smart-metal/models/mnist.json");

    // net.save_to(model_path)
    //     .expect(&format!("Failed to save to {:?}", model_path));
    net.load_from(model_path).expect("Model not found");

    let accuracy = test(&mut net, &data.images, &data.classes);
    println!("Accuracy of {} %", accuracy * 100.0);
}

fn test(net: &mut Net, data: &Vec<DMatrix<f64>>, classes: &Vec<DMatrix<f64>>) -> f64 {
    let mut res = 0.0;
    for (x, y) in data.iter().zip(classes) {
        let out = net.predict_prob(x);
        let y = one_hot_decode(y);

        if out == y {
            res += 1.0;
        }
    }

    res / data.len() as f64
}
