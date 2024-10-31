mod nn;

use crate::nn::{*, matrix::Mat, activation::Function};

const EPOCHS: u32 = 10000;
const RATE: f64 = 2.0;

fn main() {
    let data: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let labels: Vec<f64> = vec![
        0.0,
        1.0,
        1.0,
        0.0,
    ];

    let arch = vec![2, 2, 1];

    let mut net = Network::new_rand(arch, Function::SIGMOID);

    let mut input = Mat::new(1, 1);
    let mut label = Mat::new(1, 1);
    let mut out = Mat::new(1, 1);

    for i in 0..EPOCHS {
        for j in 0..labels.len() {
            input = Mat::from_vec(1, 2, data[j].clone());
            label = Mat::from_vec(1, 1, labels[j..j+1].to_vec());

            out = net.forward(&input);
            net.update(&input, &label, RATE);
        }

        let cost = net.loss(&out, &label);
        println!("{cost}");
    }

    net.show_params();

    for i in 0..4 {
        input = Mat::from_vec(1, 2, data[i].clone());
        label = Mat::from_vec(1, 1, labels[i..i+1].to_vec());
        net.show(&input, &label);
    }
}
