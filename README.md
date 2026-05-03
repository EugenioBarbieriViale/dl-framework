# Deep Learning Framework in Rust
# Compelling
## inizialize nabla with NetParams
## Implement softmax decently
## Option batches, if None don't use them
## Use Rayon for multi threadding across batches
- [https://doc.rust-lang.org/book/ch16-00-concurrency.html]

# Secondary
## Warn or stop if model.json already exists
## Improve load_data

# Maybe useless
## Load MNIST dataset from CSV file instead of whatever that is
## Use static matrices instead of dynamically allocated ones
- compare performace between the twos

# Features

# Done
## Organize in lib/
## Move code out of mod.rs and use it only to mod other functions
## Benchmarks
## Predict sees if softmax is used and one-hot-decodes the output of the net
## Remove run_training and from_model
## Introduce batches
## Change parameters inizialization, as it affects net performance and benchmarks
