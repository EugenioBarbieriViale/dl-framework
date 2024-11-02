use rand::random;

#[derive(Debug, Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub elems: Vec<Vec<f64>>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            elems: vec![vec![0.0; cols]; rows],
        }
    }

    pub fn from(elems: Vec<Vec<f64>>) -> Self {
        Self {
            rows: elems.len(),
            cols: elems[0].len(),
            elems,
        }
    }

    pub fn new_rand(rows: usize, cols: usize) -> Self {
        let mut elems: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                elems[i][j] = random();
            }
        }

        Self {
            rows,
            cols,
            elems,
        }
    }

    pub fn from_vec(rows: usize, cols: usize, elems_1d: &Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            elems: elems_1d.chunks(cols).map(|v| v.to_vec()).collect::<Vec<_>>(),
        }
    }

    pub fn from_scalar(n: f64) -> Self {
        Self {
            rows: 1,
            cols: 1,
            elems: vec![vec![n]],
        }
    }

    pub fn sum(&mut self, other: &Mat) {
        if self.rows != other.rows && self.cols != other.cols {
            panic!("Cannot sum matrix {}x{} with {}x{}", self.rows, self.cols, other.rows, other.cols);
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.elems[i][j] += other.elems[i][j];
            } 
        } 
    }

    pub fn sub(&mut self, other: &Mat) {
        if self.rows != other.rows && self.cols != other.cols {
            panic!("Cannot subtract matrix {}x{} from {}x{}", other.rows, other.cols, self.rows, self.cols);
        }

        for i in 0..self.rows {
            for j in 0..self.cols {
                self.elems[i][j] -= other.elems[i][j];
            } 
        } 
    }

    pub fn scalar_mult(&mut self, scalar: f64) {
        for i in 0..self.rows {
            for j in 0..self.cols {
                self.elems[i][j] *= scalar;
            } 
        } 
    }

    pub fn dot(&self, other: &Mat) -> Self {
        if self.cols != other.rows {
            panic!("Cannot multiply matrix {}x{} with {}x{}", self.rows, self.cols, other.rows, other.cols);
        }

        let mut ans = vec![vec![0.0; other.cols]; self.rows];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..other.rows {
                    ans[i][j] += self.elems[i][k] * other.elems[k][j];
                }
            }
        }
        Self {
            rows: self.rows,
            cols: other.cols,
            elems: ans,
        }
    }

    pub fn print(&self) {
        println!("{:?}", self);
    }
}
