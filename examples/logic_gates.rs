mod nn;

use crate::nn::{*, matrix::Mat, activation::Function};

const EPOCHS: u32 = 100000;
const RATE: f64 = 1.0;

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

    for i in 0..EPOCHS {
        let j = (i % data.len() as u32) as usize;

        let input = Mat::from_vec(1, 2, &data[j]);
        let label = Mat::from_scalar(labels[j] as f64);

        let out = net.forward(&input);
        net.update(&input, &label, RATE);

        let cost = net.loss(&out, &label);
        println!("{cost}");
    }

    net.show_params();

    for i in 0..4 {
        let input = Mat::from_vec(1, 2, &data[i]);
        let label = Mat::from_scalar(labels[i] as f64);
        net.show(&input, &label);
    }
}
