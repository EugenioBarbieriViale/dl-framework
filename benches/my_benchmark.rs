use smartmetal::net::Net;
use smartmetal::net::functions::*;
use smartmetal::net::hyperparams::Hyperparams;

use criterion::{Criterion, criterion_group, criterion_main};
use nalgebra::DMatrix;
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    let data: Vec<DMatrix<f64>> = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]
        .chunks(2)
        .map(|s| DMatrix::from_row_slice(2, 1, s))
        .collect();

    let labels: Vec<DMatrix<f64>> = [0.0, 1.0, 1.0, 0.0]
        .chunks(1)
        .map(|s| DMatrix::from_row_slice(1, 1, s))
        .collect();

    let arch = vec![2, 2, 1];

    let hypp = Hyperparams::new(100, 5e-2);
    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; 2];

    let mut net = Net::new(arch, act_funcs, loss_func);

    c.bench_function("xor training", |b| {
        b.iter(|| net.train(black_box(&data), black_box(&labels), black_box(&hypp)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
