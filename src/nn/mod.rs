pub mod matrix;
pub mod activation;

use matrix::Mat;
use activation::{Function, activate};

const H: f64 = 1e-4;

macro_rules! square {
    ($s:expr) => {
        ($s)*($s)
    }
}

pub struct Network {
    arch: Vec<usize>,
    act: Function,

    ws: Vec<Mat>,
    bs: Vec<Mat>,
    _as: Vec<Mat>,
}

impl Network {
    fn new(arch: Vec<usize>, act: Function) -> Self {
        let mut ws: Vec<Mat> = vec![];
        let mut bs: Vec<Mat> = vec![];
        let mut _as: Vec<Mat> = vec![Mat::new(0,0)];

        for l in 1..arch.len() {
            let w_l = Mat::new(arch[l-1], arch[l]);
            let b_l = Mat::new(1, arch[l]);

            ws.push(w_l);
            bs.push(b_l);
            _as.push(Mat::new(1, arch[l]));
        }

        assert!(ws.len() == bs.len());
        assert!(ws.len() == _as.len() - 1);

        Self {
            arch,
            act,
            ws,
            bs,
            _as,
        }
    }

    pub fn new_rand(arch: Vec<usize>, act: Function) -> Self {
        let mut ws: Vec<Mat> = vec![];
        let mut bs: Vec<Mat> = vec![];
        let mut _as: Vec<Mat> = vec![Mat::new(0,0)];

        for l in 1..arch.len() {
            let w_l = Mat::new_rand(arch[l-1], arch[l]);
            let b_l = Mat::new_rand(1, arch[l]);

            ws.push(w_l);
            bs.push(b_l);
            _as.push(Mat::new(1, arch[l]));
        }

        assert!(ws.len() == bs.len());
        assert!(ws.len() == _as.len() - 1);

        Self {
            arch,
            act,
            ws,
            bs,
            _as,
        }
    }

    pub fn forward(&mut self, input: &Mat) -> Mat {
        self._as[0] = input.clone();
        for l in 0..(self.arch.len() - 1) {
            let mut o_l = self._as[l].dot(&self.ws[l]);
            o_l.sum(&self.bs[l]);
            self._as[l+1] = activate(&o_l, self.act);
        }
        self._as.last().unwrap().clone()
    }

    pub fn loss(&self, out: &Mat, label: &Mat) -> f64 {
       assert!(label.cols == out.cols);
       assert!(out.rows == 1);
       assert!(label.rows == 1);

       let mut cost = 0.0;
       for i in 0..out.cols {
           cost += square!(out.elems[0][i] - label.elems[0][i]);
       }

       cost /= out.cols as f64;
       cost
    }

    fn finite_diff(&mut self, input: &Mat, label: &Mat) -> Self {
        let mut g = Self::new(self.arch.clone(), self.act);

        let prev_loss = self.loss(self._as.last().unwrap(), label);

        for k in 0..self.ws.len() {

            for i in 0..self.ws[k].rows {
                for j in 0..self.ws[k].cols {
                    self.ws[k].elems[i][j] += H;

                    let out = self.forward(input);
                    g.ws[k].elems[i][j] = (self.loss(&out, label) - prev_loss) / H;

                    self.ws[k].elems[i][j] -= H;
                }
            }
        }

        for k in 0..self.bs.len() {

            for i in 0..self.bs[k].rows {
                for j in 0..self.bs[k].cols {
                    self.bs[k].elems[i][j] += H;

                    let out = self.forward(input);
                    g.bs[k].elems[i][j] = (self.loss(&out, label) - prev_loss) / H;

                    self.bs[k].elems[i][j] -= H;
                }
            }
        }

        g
    }

    pub fn update(&mut self, input: &Mat, label: &Mat, rate: f64) {
        let mut g = self.finite_diff(input, label);
        assert!(g.arch.len() == self.arch.len());

        for l in 0..(g.arch.len()-1) {
            g.ws[l].scalar_mult(rate);
            g.bs[l].scalar_mult(rate);
        }

        for l in 0..(self.arch.len()-1) {
            self.ws[l].sub(&g.ws[l]);
            self.bs[l].sub(&g.bs[l]);
        }
    }

    pub fn show_params(&self) {
        println!("WEIGHTS: {:#?}", self.ws);
        println!("BIASES: {:#?}", self.bs);
    }

    pub fn show(&mut self, input: &Mat, label: &Mat) {
        let show_out = self.forward(&input);
        println!("{:?}: {:?} -> {:?}", input.elems, label.elems, show_out.elems);
    }
}
