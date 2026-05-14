use crate::net::buffer::Buffer;

use super::Net;
use super::functions::*;
use super::hyperparams::Hyperparams;

use nalgebra::DMatrix;
use rayon::prelude::*;

impl Net {
    pub fn forward(&self, x: &DMatrix<f64>, buff: &mut Buffer) {
        buff.activations[0] = x.clone();

        for i in 0..self.layers {
            buff.zs[i] = (&self.params.weights[i] * &buff.activations[i]) + &self.params.biases[i];
            buff.activations[i + 1] = self.act_functions[i].compute(&buff.zs[i]);
        }
    }

    fn backward(&self, y: &DMatrix<f64>, out: &DMatrix<f64>, buff: &mut Buffer) {
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
                self.act_functions[self.layers - 1].derivative(&buff.zs[self.layers - 1]);
            loss_grad.component_mul(&activation_deriv)
        };

        buff.nabla_w[self.layers - 1] = &delta * buff.activations[self.layers - 1].transpose();
        buff.nabla_b[self.layers - 1] = delta.clone();

        for l in (0..self.layers - 1).rev() {
            delta = self.params.weights[l + 1].transpose() * &delta;
            let activation_deriv = self.act_functions[l].derivative(&buff.zs[l]);
            delta = delta.component_mul(&activation_deriv);

            buff.nabla_w[l] = &delta * buff.activations[l].transpose();
            buff.nabla_b[l] = delta.clone();
        }
    }

    fn update_params(&mut self, buff: &Buffer, learning_rate: f64) {
        for l in 0..self.layers {
            self.params.weights[l] -= &buff.nabla_w[l] * learning_rate;
            self.params.biases[l] -= &buff.nabla_b[l] * learning_rate;
        }
    }

    fn batch_train(
        &self,
        cost: &mut f64,
        x_batch: &[DMatrix<f64>],
        y_batch: &[DMatrix<f64>],
        buff: &mut Buffer,
    ) {
        let res: Vec<(f64, Vec<DMatrix<f64>>, Vec<DMatrix<f64>>)> = x_batch
            .par_iter()
            .zip_eq(y_batch.par_iter())
            .map(|(xi, yi)| {
                let mut buff = Buffer::alloc(&self.arch, self.layers);

                self.forward(xi, &mut buff);

                let out = buff.activations[self.layers].clone();
                let cost = self.loss_function.compute(&out, yi);

                self.backward(yi, &out, &mut buff);
                (cost, buff.nabla_w, buff.nabla_b)
            })
            .collect();

        buff.zero_grad(self.layers);
        for (sample_cost, nw, nb) in res {
            *cost += sample_cost;
            for i in 0..nw.len() {
                buff.nabla_w[i] += &nw[i];
                buff.nabla_b[i] += &nb[i];
            }
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

        let mut buff = Buffer::alloc(&self.arch, self.layers);

        for e in 0..hypp.epochs {
            let mut cost = 0.0;
            data.chunks(hypp.batch_size)
                .zip(classes.chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    self.batch_train(&mut cost, x_batch, y_batch, &mut buff);
                    self.update_params(&buff, hypp.learning_rate / hypp.batch_size as f64);
                });
            cost /= len as f64;
            // println!("Epoch {e}: loss = {}", *cost.lock().unwrap());
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

        let mut buff = Buffer::alloc(&self.arch, self.layers);

        for e in 0..hypp.epochs {
            let mut cost = 0.0;
            data.chunks(hypp.batch_size)
                .zip(classes.chunks(hypp.batch_size))
                .for_each(|(x_batch, y_batch)| {
                    buff.zero_grad(self.layers);

                    for (xi, yi) in x_batch.iter().zip(y_batch.iter()) {
                        self.forward(xi, &mut buff);

                        let out = buff.activations[self.layers].clone();
                        cost += self.loss_function.compute(&out, yi);

                        self.backward(yi, &out, &mut buff);
                    }

                    self.update_params(&buff, hypp.learning_rate / hypp.batch_size as f64);
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

        let mut buff = Buffer::alloc(&self.arch, self.layers);

        for e in 0..hypp.epochs {
            let mut cost = 0.0;
            data.into_iter().zip(classes).for_each(|(x, y)| {
                buff.zero_grad(self.layers);
                self.forward(x, &mut buff);

                let out = buff.activations[self.layers].clone();
                cost += self.loss_function.compute(&out, y);

                self.backward(y, &out, &mut buff);

                self.update_params(&buff, hypp.learning_rate / hypp.batch_size as f64);
            });
            cost /= len as f64;
            // println!("Epoch {e}: loss = {}", cost);
        }
    }
}
