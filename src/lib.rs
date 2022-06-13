use thiserror::Error;
use nalgebra::{Matrix, Dim, RawStorageMut};
use core::ops::{AddAssign, Mul, MulAssign};

#[cfg(any(doc, test, doctest))]
extern crate num_rational;
#[cfg(any(doc, test, doctest))]
extern crate num_bigint;

pub struct MatrixReprOfLinEq<T,R,C,S>(pub Matrix<T,R,C,S>);

#[derive(Error, Debug)]
pub enum BinaryRowIdxOutOfBoundsError {
    #[error("First row index is out of bounds: {0:?}")]
    FirstIdxOutOfBounds((usize, usize)),
    #[error("Second row index is out of bounds: {0:?}")]
    SecondIdxOutOfBounds((usize, usize)),
    #[error("Both row indices are out of bounds: {0:?}")]
    BothIdcesOutOfBounds((usize, usize)),
}

#[derive(Error, Debug)]
#[error("Row index is out of bounds: {0:?}")]
pub struct RowIdxOutOfBoundsError(usize);

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
    /// Unlike [`nalgebra::base::Matrix::swap_rows`], this method doesn't require the entries
    /// to implement [`nalgebra::base::Scalar`].
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it does not check if the indices are valid.
    pub unsafe fn row_xchg_unchecked(&mut self, i_1: usize, i_2: usize)
    {
        let ncols = self.0.ncols();
        (0..ncols)
            .map(|j| ((i_1, j), (i_2, j)))
            .for_each(|(row_col1, row_col2)| {
                self.0.swap_unchecked(row_col1, row_col2);
            }
        );
    }

    pub fn row_xchg(&mut self, i_1: usize, i_2: usize) -> Result<(), BinaryRowIdxOutOfBoundsError> {
        use BinaryRowIdxOutOfBoundsError::*;

        let nrows = self.0.nrows();
        match (i_1,i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            _ => {
                Ok(unsafe { self.row_xchg_unchecked(i_1, i_2) })
            }
        }
    }
}

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
where
    T: Clone + AddAssign,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T,R,C>,
{
    pub unsafe fn row_add_unchecked<'a,'b>(&'a mut self, i_1: usize, i_2: usize, factor: &'b T)
    where
        T: Mul<&'b T, Output=T> + 'b + AddAssign<T>,
    {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            let corresponding_entry = self.0[(i_2, j)].to_owned();
            *self.0.get_unchecked_mut((i_1, j)) += corresponding_entry * factor;
        }
    }


    pub fn row_add<'a,'b>(&'a mut self, i_1: usize, i_2: usize, factor: &'b T) -> Result<(), BinaryRowIdxOutOfBoundsError>
    where
        T: Mul<&'b T, Output=T> + 'b + AddAssign<T>,
    {
        use BinaryRowIdxOutOfBoundsError::*;

        let nrows = self.0.nrows();
        match (i_1,i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            _ => {
                Ok(unsafe { self.row_add_unchecked(i_1, i_2, factor) })
            }
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
    pub unsafe fn row_mul_unchecked<'a>(&mut self, i: usize, factor: &'a T)
    where
        T: MulAssign<&'a T>,
    {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            *self.0.get_unchecked_mut((i, j)) *= factor;
        }
    }

    pub fn row_mul<'a>(&mut self, i: usize, factor: &'a T) -> Result<(), RowIdxOutOfBoundsError>
    where
        T: MulAssign<&'a T>,
    {
        let nrows = self.0.nrows();
        if i >= nrows {
            Err(RowIdxOutOfBoundsError(i))
        } else {
            Ok(unsafe { self.row_mul_unchecked(i, factor) })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    #[test]
    fn row_xchg_works_for_prim_ints() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1usize, 2usize;
            3usize, 4usize;
        ));
        unsafe { m.row_xchg_unchecked(0, 1) };
        assert_eq!(
            m.0,
            matrix!(
                3usize, 4usize;
                1usize, 2usize;
            )
        );
    }

    #[test]
    fn row_add_works_for_prim_ints() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1i32, 2i32;
            3i32, 4i32;
        ));
        unsafe { m.row_add_unchecked(0, 1, &1i32) };
        assert_eq!(
            m.0,
            matrix!(
                4i32, 6i32;
                3i32, 4i32;
            )
        );
    }

    #[test]
    fn row_mul_works_for_prim_ints() {
        let mut m = MatrixReprOfLinEq::new(matrix!(
            1i32, 2i32;
            3i32, 4i32;
        ));
        unsafe { m.row_mul_unchecked(0, &2i32) };
        assert_eq!(
            m.0,
            matrix!(
                2i32, 4i32;
                3i32, 4i32;
            )
        );
    }

    #[test]
    fn row_xchg_works_with_num_rational_entries(){
        use num_rational::BigRational;
        use num_bigint::BigInt;
        
        // 1/2 3/4
        // 5/6 7/8
        let mut m = MatrixReprOfLinEq::new(matrix!(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(3), BigInt::from(4));
            BigRational::new(BigInt::from(5), BigInt::from(6)),
            BigRational::new(BigInt::from(7), BigInt::from(8));
        ));
        unsafe { m.row_xchg_unchecked(0, 1) };
        // 5/6 7/8
        // 1/2 3/4
        assert_eq!(
            m.0,
            matrix!(
                BigRational::new(BigInt::from(5), BigInt::from(6)),
                BigRational::new(BigInt::from(7), BigInt::from(8));
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::new(BigInt::from(3), BigInt::from(4));
            )
        );
    }

    #[test]
    fn row_add_works_with_num_rational_entries(){
        use num_rational::BigRational;
        use num_bigint::BigInt;
        
        // 1/2 3/4
        // 5/6 7/8
        let mut m = MatrixReprOfLinEq::new(matrix!(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(3), BigInt::from(4));
            BigRational::new(BigInt::from(5), BigInt::from(6)),
            BigRational::new(BigInt::from(7), BigInt::from(8));
        ));
        // 1/1 = 1
        let factor = BigRational::new(BigInt::from(1), BigInt::from(1));
        // Adds 0th row to 1st row with factor
        unsafe { m.row_add_unchecked(1, 0, &factor) };
        // 1/2               3/4
        // (1/2 + 5/6 * 1/1) (3/4 + 7/8 * 1/1)
        //
        // or
        //
        // 1/2         3/4
        // (3/6 + 5/6) (6/8 + 7/8)
        //
        // or
        //
        // 1/2 3/4
        // 8/6 13/8
        assert_eq!(
            m.0,
            matrix!(
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::new(BigInt::from(3), BigInt::from(4));
                BigRational::new(BigInt::from(8), BigInt::from(6)),
                BigRational::new(BigInt::from(13), BigInt::from(8));
            )
        );
    }

    #[test]
        fn row_mul_works_with_num_rational_entries(){
            use num_rational::BigRational;
            use num_bigint::BigInt;
            
            // 1/2 3/4
            // 5/6 7/8
            let mut m = MatrixReprOfLinEq::new(matrix!(
                BigRational::new(BigInt::from(1), BigInt::from(2)),
                BigRational::new(BigInt::from(3), BigInt::from(4));
                BigRational::new(BigInt::from(5), BigInt::from(6)),
                BigRational::new(BigInt::from(7), BigInt::from(8));
            ));
            // 2/1 = 2
            let factor = BigRational::new(BigInt::from(2), BigInt::from(1));
            // Multiplies 0th row by factor
            unsafe { m.row_mul_unchecked(0, &factor) };
            // 1/2 * 2/1 = 2/2 = 1/1
            // 3/4 * 2/1 = 6/4 = 3/2
            assert_eq!(
                m.0,
                matrix!(
                    BigRational::new(BigInt::from(1), BigInt::from(1)),
                    BigRational::new(BigInt::from(3), BigInt::from(2));
                    BigRational::new(BigInt::from(5), BigInt::from(6)),
                    BigRational::new(BigInt::from(7), BigInt::from(8));
                )
            );
        }
}