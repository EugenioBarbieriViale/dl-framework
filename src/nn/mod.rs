pub mod matrix;
pub mod activation;

use matrix::Mat;
use activation::{Function, activate};

const H: f64 = 1e-2;

macro_rules! square {
    ($s:expr) => {
        ($s)*($s)
    }
}

pub struct Network {
    act: Function,

    w1: Mat,
    b1: Mat,

    w2: Mat,
    b2: Mat,

    pub out: Mat,
}

impl Network {
    pub fn new(act: Function) -> Self {
        Self {
            act,

            w1: Mat::new(2, 2),
            b1: Mat::new(1, 2),

            w2: Mat::new(2, 1),
            b2: Mat::new(1, 1),

            out: Mat::new(2, 1),
        }
    }

    pub fn new_rand(act: Function) -> Self {
        Self {
            act,

            w1: Mat::new_rand(2, 2),
            b1: Mat::new_rand(1, 2),

            w2: Mat::new_rand(2, 1),
            b2: Mat::new_rand(1, 1),

            out: Mat::new(2, 1),
        }
    }

    pub fn forward(&self, input: &Mat) -> Mat {
        let o1 = &mut input.dot(&self.w1);
        o1.sum(&self.b1);
        let a1 = activate(o1, self.act);

        let o2 = &mut a1.dot(&self.w2);
        o2.sum(&self.b2);

        activate(o2, self.act)
    }

    pub fn loss(&self, out: &Mat, label: &Mat) -> f64 {
       if label.cols != out.cols || out.rows != 1 || label.rows != 1 {
           panic!("Output or labels vector is not of the right dimensions");
       }

       let mut cost = 0.0;
       for i in 0..out.cols {
           cost += square!(out.elems[0][i] - label.elems[0][i]);
       }
       cost /= out.cols as f64;
       cost
    }

    // TODO clone self, not mut
    fn finite_diff(&mut self, input: &Mat, label: &Mat) -> Self {
        let mut g = Self::new(self.act);
        let prev_loss = self.loss(&self.out, label);

        for i in 0..self.w1.rows {
            for j in 0..self.w1.cols {
                self.w1.elems[i][j] += H;
                g.w1.elems[i][j] = (self.loss(&self.forward(input), label) - prev_loss) / H;
                self.w1.elems[i][j] -= H;
            }
        }

        for i in 0..self.b1.rows {
            for j in 0..self.b1.cols {
                self.b1.elems[i][j] += H;
                g.b1.elems[i][j] = (self.loss(&self.forward(input), label) - prev_loss) / H;
                self.b1.elems[i][j] -= H;
            }
        }

        for i in 0..self.w2.rows {
            for j in 0..self.w2.cols {
                self.w2.elems[i][j] += H;
                g.w2.elems[i][j] = (self.loss(&self.forward(input), label) - prev_loss) / H;
                self.w2.elems[i][j] -= H;
            }
        }

        for i in 0..self.b2.rows {
            for j in 0..self.b2.cols {
                self.b2.elems[i][j] += H;
                g.b2.elems[i][j] = (self.loss(&self.forward(input), label) - prev_loss) / H;
                self.b2.elems[i][j] -= H;
            }
        }
        g
    }

    pub fn update(&mut self, input: &Mat, label: &Mat, rate: f64) {
        let mut g = self.finite_diff(input, label);

        g.w1.scalar_mult(rate);
        g.b1.scalar_mult(rate);
                              
        g.w2.scalar_mult(rate);
        g.b2.scalar_mult(rate);

        self.w1.sub(&g.w1);
        self.b1.sub(&g.b1);

        self.w2.sub(&g.w2);
        self.b2.sub(&g.b2);
    }

    pub fn show_net(&self, input: &Mat, label: &Mat) {
        let show_out = self.forward(&input);
        println!("{:?}: {:?} -> {:?}", input.elems, label.elems, show_out.elems);
    }
}
