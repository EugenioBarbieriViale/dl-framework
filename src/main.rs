use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::Hyperparams;
use smartmetal::net::load_mnist::{MnistDataset, load_data, one_hot_decode};

use std::path::Path;

fn main() {
    let model_path = Path::new("/home/eu/programming/smart-metal/models/mnist.json");

    let data =
        load_data("/home/eu/programming/smart-metal/data/train").expect("Could not load dataset.");
    println!(
        "MNIST dataset has been loaded ({} images found).",
        data.images.len()
    );

    let arch = vec![28 * 28, 512, 512, 10];

    let params = Hyperparams::new(5, 1e-2);
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 2];
    act_funcs.push(ActivationFunction::Softmax);

    let mut net = Net::new(arch, act_funcs, loss_func);
    println!("Neural network has been initialized\n");

    // run_training(&mut net, &data, &params, model_path);
    from_model(&mut net, &data, model_path);
}

#[allow(unused)]
fn run_training(net: &mut Net, data: &MnistDataset, params: &Hyperparams, path: &Path) {
    println!("Training has started...");
    net.train(&data.images, &data.classes, &params);
    println!("Training ended.");

    println!("Saving model...");
    net.save_to(path)
        .expect(&format!("Failed to save to {:?}", path));
    println!("Done.");
}

#[allow(unused)]
fn from_model(net: &mut Net, data: &MnistDataset, path: &Path) {
    println!("Loading model...");
    let p = net.load_from(path).unwrap();
    println!("Done");

    net.params = p;

    let idx = 4440;
    let out = net.predict(&data.images[idx]);
    println!("Prediction: {}", one_hot_decode(&out));
    println!("Actual value: {}", one_hot_decode(&data.classes[idx]));
}
