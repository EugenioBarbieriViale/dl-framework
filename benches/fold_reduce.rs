use smartmetal::net::Net;
use smartmetal::net::functions::*;
use smartmetal::net::hyperparams::*;
use smartmetal::net::init::*;

use nalgebra::DMatrix;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    let len = 28 * 28 * 100;
    let batch_size = 64;
    let epochs = 2;

    let arch = vec![28 * 28, 512, 512, 10];

    let input_size = arch[0];
    let output_size = *arch.last().unwrap();

    let loss_func = LossFunction::SquaredError;
    let act_funcs = vec![ActivationFunction::ReLU; arch.len() - 1];
    let init = Initialization::Random;
    let hypp = Hyperparams::new(epochs, batch_size, 5e-2);

    let range: Vec<f64> = (0..len).map(|x| x as f64 / len as f64).collect();

    let data: Vec<DMatrix<f64>> = range
        .chunks(input_size)
        .map(|s| DMatrix::from_row_slice(input_size, 1, s))
        .collect();

    let labels: Vec<DMatrix<f64>> = range[..len / input_size * output_size]
        .to_vec()
        .chunks(output_size)
        .map(|s| DMatrix::from_row_slice(output_size, 1, s))
        .collect();

    let mut group = c.benchmark_group("fold_reduce");
    group.sample_size(10);

    group.bench_with_input(BenchmarkId::new("Normal", epochs), &epochs, |b, _| {
        b.iter(|| {
            let mut net = Net::new(arch.clone(), act_funcs.clone(), loss_func.clone(), &init);
            net.par_train(black_box(&data), black_box(&labels), black_box(&hypp))
        })
    });

    group.bench_with_input(BenchmarkId::new("For real", epochs), &epochs, |b, _| {
        b.iter(|| {
            let mut net = Net::new(arch.clone(), act_funcs.clone(), loss_func.clone(), &init);
            net.par_train_fr(black_box(&data), black_box(&labels), black_box(&hypp))
        })
    });

    group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
