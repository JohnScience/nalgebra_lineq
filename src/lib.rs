use nalgebra::{Matrix, Dim};

pub struct MatrixReprOfLinEq<T,R,C,S>(pub Matrix<T,R,C,S>);

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S> {
    pub fn new(matrix: Matrix<T,R,C,S>) -> Self {
        MatrixReprOfLinEq(matrix)
    }

    pub fn to_matrix(self) -> Matrix<T,R,C,S> {
        self.0
    }
}

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
{
    /// Switches rows `i_1` and `i_2` in the matrix.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it does not check if the indices are valid.
    unsafe fn switch_rows(&mut self, i_1: usize, i_2: usize) {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            let tmp = self.0[(i_1, j)];
            self.0[(i_1, j)] = self.0[(i_2, j)];
            self.0[(i_2, j)] = tmp;
        }
    }
}