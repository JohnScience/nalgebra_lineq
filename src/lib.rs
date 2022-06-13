use nalgebra::{Matrix, Dim, RawStorageMut};
use core::ops::{AddAssign, Mul, MulAssign};

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
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T,R,C>
{
    /// Switches rows `i_1` and `i_2` in the matrix.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it does not check if the indices are valid.
    pub unsafe fn row_xchg(&mut self, i_1: usize, i_2: usize)
    {
        let ncols = self.0.ncols();
        (0..ncols)
            .map(|j| ((i_1, j), (i_2, j)))
            .for_each(|(row_col1, row_col2)| {
                self.0.swap_unchecked(row_col1, row_col2);
            }
        );
    }
}

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
where
    T: Clone + AddAssign,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T,R,C>,
{
    pub unsafe fn row_add<'a,'b>(&'a mut self, i_1: usize, i_2: usize, factor: &'b T)
    where
        T: Mul<&'b T, Output=T> + 'b + AddAssign<T>,
    {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            let corresponding_entry = self.0[(i_2, j)].to_owned();
            self.0[(i_1, j)] += corresponding_entry * factor;
        }
    }
}

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
where
    T: Clone + MulAssign,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T,R,C>,
{
    pub unsafe fn row_mul<'a>(&mut self, i: usize, factor: &'a T)
    where
        T: MulAssign<&'a T>,
    {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            self.0[(i, j)] *= factor;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    #[test]
    fn row_xchg_works() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1usize, 2usize;
            3usize, 4usize;
        ));
        unsafe { m.row_xchg(0, 1) };
        assert_eq!(
            m.0,
            matrix!(
                3usize, 4usize;
                1usize, 2usize;
            )
        );
    }

    #[test]
    fn row_add_works() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1i32, 2i32;
            3i32, 4i32;
        ));
        unsafe { m.row_add(0, 1, &1i32) };
        assert_eq!(
            m.0,
            matrix!(
                4i32, 6i32;
                3i32, 4i32;
            )
        );
    }

    #[test]
    fn row_mul_works() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1i32, 2i32;
            3i32, 4i32;
        ));
        unsafe { m.row_mul(0, &2i32) };
        assert_eq!(
            m.0,
            matrix!(
                2i32, 4i32;
                3i32, 4i32;
            )
        );
    }
}