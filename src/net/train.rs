use super::Net;
use super::functions::*;
use super::hyperparams::Hyperparams;

use nalgebra::DMatrix;
use rayon::prelude::*;

impl Net {
    pub fn forward(&mut self, x: &DMatrix<f64>) -> DMatrix<f64> {
        self.activations[0] = x.clone(); // slow

        for i in 0..self.layers {
            self.zs[i] = (&self.params.weights[i] * &self.activations[i]) + &self.params.biases[i];
            self.activations[i + 1] = self.act_functions[i].compute(&self.zs[i]);
        }

        // self.activations.last().unwrap().clone() // slow
        self.activations[self.layers].clone()
    }

    fn backward(
        &self,
        y: &DMatrix<f64>,
        nabla_w: &mut Vec<DMatrix<f64>>,
        nabla_b: &mut Vec<DMatrix<f64>>,
    ) {
        let out = &self.activations[self.layers];
        let loss_grad = self.loss_function.gradient(out, y);

        let mut delta = if matches!(
            self.act_functions[self.layers - 1],
            ActivationFunction::Softmax
        ) {
            // compute softmax
            let s = out;
            let dot: f64 = s.component_mul(&loss_grad).sum();
            s.component_mul(&(&loss_grad - DMatrix::from_element(s.nrows(), 1, dot)))
        } else {
            let activation_deriv =
                self.act_functions[self.layers - 1].derivative(&self.zs[self.layers - 1]);
            loss_grad.component_mul(&activation_deriv)
        };

        nabla_w[self.layers - 1] += &delta * self.activations[self.layers - 1].transpose();
        nabla_b[self.layers - 1] += delta.clone();

        for l in (0..self.layers - 1).rev() {
            delta = self.params.weights[l + 1].transpose() * &delta;
            let activation_deriv = self.act_functions[l].derivative(&self.zs[l]);
            delta = delta.component_mul(&activation_deriv);

            nabla_w[l] += &delta * self.activations[l].transpose();
            nabla_b[l] += delta.clone();
        }
    }

    fn update_params(
        &mut self,
        nabla_w: &Vec<DMatrix<f64>>,
        nabla_b: &Vec<DMatrix<f64>>,
        learning_rate: f64,
    ) {
        for l in 0..self.layers {
            self.params.weights[l] -= &nabla_w[l] * learning_rate;
            self.params.biases[l] -= &nabla_b[l] * learning_rate;
        }
    }

    pub fn seq_train(
        &mut self,
        data: &Vec<DMatrix<f64>>,
        classes: &Vec<DMatrix<f64>>,
        hypp: &Hyperparams,
    ) {
        let len = data.len();
        assert_eq!(len, classes.len());

        for e in 0..hypp.epochs {
            self.cost = 0.0;
            data.chunks(hypp.batch_size)
                .zip(classes.chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    let (mut nabla_w, mut nabla_b) = self.init_gradients();
                    for (xi, yi) in x_batch.iter().zip(y_batch.iter()) {
                        let _ = &self.forward(xi);

                        let out = &self.activations[self.layers];
                        self.cost += self.loss_function.compute(out, yi);

                        self.backward(yi, &mut nabla_w, &mut nabla_b);
                    }
                    self.update_params(
                        &nabla_w,
                        &nabla_b,
                        hypp.learning_rate / hypp.batch_size as f64,
                    );
                });
            self.cost /= len as f64;
            println!("Epoch {e}: loss = {}", self.cost);
        }
    }

    pub fn par_train(
        &mut self,
        data: &Vec<DMatrix<f64>>,
        classes: &Vec<DMatrix<f64>>,
        hypp: &Hyperparams,
    ) {
        let len = data.len();
        assert_eq!(len, classes.len());

        for e in 0..hypp.epochs {
            self.cost = 0.0;
            data.par_iter()
                .chunks(hypp.batch_size)
                .zip(classes.par_iter().chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    let (mut nabla_w, mut nabla_b) = self.init_gradients();
                    for (xi, yi) in x_batch.iter().zip(y_batch.iter()) {
                        let _ = &self.forward(xi);

                        let out = &self.activations[self.layers];
                        self.cost += self.loss_function.compute(out, yi);

                        self.backward(yi, &mut nabla_w, &mut nabla_b);
                    }
                    self.update_params(
                        &nabla_w,
                        &nabla_b,
                        hypp.learning_rate / hypp.batch_size as f64,
                    );
                });
            self.cost /= len as f64;
            println!("Epoch {e}: loss = {}", self.cost);
        }
    }
}
