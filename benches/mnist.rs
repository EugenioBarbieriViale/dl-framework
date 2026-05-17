use smartmetal::net::Net;
use smartmetal::net::functions::*;
use smartmetal::net::hyperparams::*;
use smartmetal::net::init::*;
use smartmetal::net::load_mnist::*;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;

pub fn criterion_benchmark(c: &mut Criterion) {
    let data =
        load_data("/home/eu/programming/smart-metal/data/train").expect("Could not load dataset.");
    println!(
        "MNIST dataset has been loaded ({} images found).",
        data.images.len()
    );

    println!("Benching over the mnist dataset");

    let arch = vec![28 * 28, 512, 512, 10];

    let hypp = Hyperparams::new(1, 64, 1e-2);
    let loss_func = LossFunction::CrossEntropy;
    let mut act_funcs = vec![ActivationFunction::ReLU; 2];
    act_funcs.push(ActivationFunction::Softmax);
    let init = Initialization::Kaiming(Fan::In);

    let mut group = c.benchmark_group("large_batch_size");
    group.sample_size(10);

    let mut net1 = Net::new(arch.clone(), act_funcs.clone(), loss_func.clone(), &init);
    group.bench_with_input(
        BenchmarkId::new("Sequential", hypp.epochs),
        &hypp.epochs,
        |b, _| {
            b.iter(|| {
                net1.batch_seq_train(
                    black_box(&data.images),
                    black_box(&data.classes),
                    black_box(&hypp),
                )
            })
        },
    );

    let mut net2 = Net::new(arch.clone(), act_funcs.clone(), loss_func.clone(), &init);
    group.bench_with_input(
        BenchmarkId::new("Parallel", hypp.epochs),
        &hypp.epochs,
        |b, _| {
            b.iter(|| {
                net2.par_train(
                    black_box(&data.images),
                    black_box(&data.classes),
                    black_box(&hypp),
                )
            })
        },
    );

    group.finish();

    println!("Testing last sequential model");
    let accuracy = net1.test(&data.images, &data.classes);
    println!("Accuracy of {} %", accuracy * 100.0);

    println!("Testing last parallel model");
    let accuracy = net2.test(&data.images, &data.classes);
    println!("Accuracy of {} %", accuracy * 100.0);
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
