use rand::random;

#[derive(Debug)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub elems: Vec<Vec<f64>>,
}

impl Mat {
    pub fn new(rows: usize, cols: usize) -> Self {
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

    pub fn from(rows: usize, cols: usize, elems_1d: Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            elems: elems_1d.chunks(cols).map(|v| v.to_vec()).collect::<Vec<_>>(),
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
            panic!("Cannot subtract matrix {}x{} and {}x{}", self.rows, self.cols, other.rows, other.cols);
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
