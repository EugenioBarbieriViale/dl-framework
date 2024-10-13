use rand::random;

pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    arr: Vec<Vec<f64>>,
}

impl Mat {
    pub fn init(rows: usize, cols: usize) -> Self {
        let mut arr: Vec<Vec<f64>> = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                arr[i][j] = random();
            }
        }
        Self {
            rows,
            cols,
            arr,
        }
    }

    pub fn from(rows: usize, cols: usize, arr_1d: Vec<f64>) -> Self {
        Self {
            rows,
            cols,
            arr: arr_1d.chunks(cols).map(|v| v.to_vec()).collect::<Vec<_>>(),
        }
    }

    pub fn dot(&self, other: &Mat) -> Self {
        let mut ans = vec![vec![0.0; other.rows]; self.cols];
        for i in 0..self.rows {
            for j in 0..other.cols {
                for k in 0..other.rows {
                    ans[i][j] += self.arr[i][k] * other.arr[k][j];
                }
            }
        }
        Self {
            rows: self.cols,
            cols: other.rows,
            arr: ans,
        }
    }

    pub fn print(&self) {
        println!("-----------------------");
        for r in 0..self.arr.len() {
            for c in 0..self.arr[r].len() {
                print!("{}, ", self.arr[r][c]);
            }
            println!("");
        }
        println!("-----------------------");
    }
}
