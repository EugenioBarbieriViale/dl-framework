use smartmetal::net::Net;
use smartmetal::net::functions::{ActivationFunction, LossFunction};
use smartmetal::net::hyperparams::{Fan, Hyperparams, Initialization};
use smartmetal::net::load_mnist::{load_data, one_hot_decode};

use std::path::Path;

fn main() {
    let model_path = Path::new("/home/eu/programming/smart-metal/models/mnist_test.json");

    let data =
        load_data("/home/eu/programming/smart-metal/data/train").expect("Could not load dataset.");
    println!(
        "MNIST dataset has been loaded ({} images found).",
        data.images.len()
    );

    let arch = vec![28 * 28, 512, 512, 10];

    let params = Hyperparams::new(1, 32, 1e-2);
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 2];
    act_funcs.push(ActivationFunction::Softmax);
    let init = Initialization::Kaiming(Fan::In);

    let mut net = Net::new(arch, act_funcs, loss_func, &init);
    println!("Neural network has been initialized\n");

    // net.seq_train(&data.images, &data.classes, &params);
    // net.save_to(model_path)
    //     .expect(&format!("Failed to save to {:?}", model_path));
    net.load_from(model_path).expect("Model not found");

    let idx = 1098;
    let out = net.predict_prob(&data.images[idx]);
    println!("Prediction: {}", &out);
    println!("Actual value: {}", one_hot_decode(&data.classes[idx]));
}
