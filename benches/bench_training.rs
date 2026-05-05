use criterion::BenchmarkId;
use smartmetal::net::Net;
use smartmetal::net::functions::*;
use smartmetal::net::hyperparams::*;

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

    let arch = vec![2, 256, 256, 1];

    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; 3];
    let init = Initialization::Random;

    let mut net = Net::new(arch, act_funcs, loss_func, &init);

    let mut group = c.benchmark_group("xor_comparsion");
    // for epochs in [100, 200, 500].iter() {
    for epochs in [100].iter() {
        let hypp = Hyperparams::new(*epochs, 1, 5e-2);
        group.bench_with_input(BenchmarkId::new("Sequential", epochs), epochs, |b, _| {
            b.iter(|| net.batch_seq_train(black_box(&data), black_box(&labels), black_box(&hypp)))
        });
        group.bench_with_input(BenchmarkId::new("Parallel", epochs), epochs, |b, _| {
            b.iter(|| net.par_train(black_box(&data), black_box(&labels), black_box(&hypp)))
        });
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
