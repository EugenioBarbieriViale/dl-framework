use super::Net;
use super::functions::*;
use super::hyperparams::Hyperparams;

use nalgebra::DMatrix;
use rayon::prelude::*;

impl Net {
    pub fn forward(&self, x: &DMatrix<f64>) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let mut activations = Vec::with_capacity(self.layers + 1);
        let mut zs = Vec::with_capacity(self.layers);

        for i in 0..self.layers {
            activations.push(DMatrix::zeros(
                self.params.weights[i].nrows(),
                self.params.weights[i].ncols(),
            ));
            zs.push(DMatrix::zeros(
                self.params.biases[i].nrows(),
                self.params.biases[i].ncols(),
            ));
        }
        activations.push(DMatrix::zeros(
            self.params.weights[1].nrows(),
            self.params.weights[1].ncols(),
        ));

        activations[0] = x.clone();

        for i in 0..self.layers {
            zs[i] = (&self.params.weights[i] * &activations[i]) + &self.params.biases[i];
            activations[i + 1] = self.act_functions[i].compute(&zs[i]);
        }

        (activations, zs)
    }

    fn backward(
        &self,
        y: &DMatrix<f64>,
        activations: &Vec<DMatrix<f64>>,
        zs: &Vec<DMatrix<f64>>,
    ) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let (mut nabla_w, mut nabla_b) = self.init_gradients();

        let out = &activations[self.layers];
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
                self.act_functions[self.layers - 1].derivative(&zs[self.layers - 1]);
            loss_grad.component_mul(&activation_deriv)
        };

        nabla_w[self.layers - 1] = &delta * activations[self.layers - 1].transpose();
        nabla_b[self.layers - 1] = delta.clone();

        for l in (0..self.layers - 1).rev() {
            delta = self.params.weights[l + 1].transpose() * &delta;
            let activation_deriv = self.act_functions[l].derivative(&zs[l]);
            delta = delta.component_mul(&activation_deriv);

            nabla_w[l] = &delta * activations[l].transpose();
            nabla_b[l] = delta.clone();
        }

        (nabla_w, nabla_b)
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

    fn batch_train(
        &self,
        cost: &mut f64,
        x_batch: &[DMatrix<f64>],
        y_batch: &[DMatrix<f64>],
    ) -> (Vec<DMatrix<f64>>, Vec<DMatrix<f64>>) {
        let res: Vec<(f64, Vec<DMatrix<f64>>, Vec<DMatrix<f64>>)> = x_batch
            .par_iter()
            .zip_eq(y_batch.par_iter())
            .map(|(xi, yi)| {
                let (activations, zs) = self.forward(xi);

                let out = &activations[self.layers];
                let cost = self.loss_function.compute(out, yi);

                let (nw, nb) = self.backward(yi, &activations, &zs);
                (cost, nw, nb)
            })
            .collect();

        let (mut nabla_w, mut nabla_b) = self.init_gradients();
        for (sample_cost, nw, nb) in res {
            *cost += sample_cost;
            for i in 0..nw.len() {
                nabla_w[i] += &nw[i];
                nabla_b[i] += &nb[i];
            }
        }

        (nabla_w, nabla_b)
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
            let mut cost = 0.0;
            data.chunks(hypp.batch_size)
                .zip(classes.chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    let (nabla_w, nabla_b) = self.batch_train(&mut cost, x_batch, y_batch);
                    self.update_params(
                        &nabla_w,
                        &nabla_b,
                        hypp.learning_rate / hypp.batch_size as f64,
                    );
                });
            cost /= len as f64;
            // println!("Epoch {e}: loss = {}", cost);
        }
    }

    pub fn batch_seq_train(
        &mut self,
        data: &Vec<DMatrix<f64>>,
        classes: &Vec<DMatrix<f64>>,
        hypp: &Hyperparams,
    ) {
        let len = data.len();
        assert_eq!(len, classes.len());

        for e in 0..hypp.epochs {
            let mut cost = 0.0;
            data.chunks(hypp.batch_size)
                .zip(classes.chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    let (mut nabla_w, mut nabla_b) = self.init_gradients();
                    for (xi, yi) in x_batch.iter().zip(y_batch.iter()) {
                        let (activations, zs) = self.forward(xi);

                        let out = &activations[self.layers];
                        cost += self.loss_function.compute(out, yi);

                        let (nw, nb) = self.backward(yi, &activations, &zs);
                        for i in 0..nw.len() {
                            nabla_w[i] += &nw[i];
                            nabla_b[i] += &nb[i];
                        }
                    }
                    self.update_params(
                        &nabla_w,
                        &nabla_b,
                        hypp.learning_rate / hypp.batch_size as f64,
                    );
                });
            cost /= len as f64;
            // println!("Epoch {e}: loss = {}", cost);
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
            let mut cost = 0.0;
            data.into_iter().zip(classes).for_each(|(x, y)| {
                let (mut nabla_w, mut nabla_b) = self.init_gradients();

                let _ = &self.forward(x);

                let out = &self.activations[self.layers];
                cost += self.loss_function.compute(out, y);

                self.backward(y, &mut nabla_w, &mut nabla_b);

                self.update_params(&nabla_w, &nabla_b, hypp.learning_rate);
            });
            cost /= len as f64;
            // println!("Epoch {e}: loss = {}", cost);
        }
    }
}
