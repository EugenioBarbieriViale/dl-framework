mod nn;

use crate::nn::{*, matrix::Mat};

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

    let mut l1 = Mat::new(2, 2);
    let mut l2 = Mat::new(2, 1);

    for i in 0..EPOCHS {
        for j in 0..(labels.len()-1) {
            let label = Mat::from(1, 1, labels[j..j+1].to_vec());
            let input = Mat::from(1, 2, data[j..=j+1].to_vec());

            let out = forward(&input, &l1, &l2);

            let g1 = finite_diff1(&input, &out, &label, &mut l1, &l2);
            let g2 = finite_diff2(&input, &out, &label, &l1, &mut l2);

            l1.sum(&g1);
            l2.sum(&g2);

            let c = loss(&out, &label);
            println!("{:?}", c);
        }
    }
}
