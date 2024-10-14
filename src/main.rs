mod nn;
use crate::nn::*;

const EPOCHS: u32 = 100;

fn main() {
    let data: Vec<f64> = vec![
        0.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        1.0, 1.0,
    ];

    let labels: Vec<f64> = vec![
        0.0,
        0.0,
        0.0,
        1.0,
    ];

    let label = Mat::from(1, 1, labels[0..1].to_vec());

    let input = Mat::from(1, 2, data[0..=1].to_vec());

    let l1 = Mat::init(2, 2);
    let l2 = Mat::init(2, 1);

    for i in 0..EPOCHS {
        let out = forward(&input, &l1, &l2);
        let c = loss(&out, &label);
        println!("{:?}", c);
    }
}
