mod nn;
use crate::nn::*;

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

    let input = Mat::from(4, 2, data);

    let l1 = Mat::init(2, 2);
    let l2 = Mat::init(2, 1);
}
