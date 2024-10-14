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

    let input = Mat::from(2, 4, data);
    let label = Mat::from(1, 4, labels);

    let l1 = Mat::init(2, 2);
    let l2 = Mat::init(2, 1);

    let out = forward(&input, &l1, &l2);
    println!("{}x{}", out.rows, out.cols);
    println!("{}x{}", label.rows, label.cols);
    let cost = loss(&out, &label);

    println!("{:?}", cost);
}
