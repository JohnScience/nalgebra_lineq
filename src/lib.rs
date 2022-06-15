#[cfg(any(test, doctest))]
extern crate num_bigint;
#[cfg(any(test, doctest))]
extern crate num_rational;

pub mod err;
pub mod row_add;
pub mod row_mul;
pub mod row_xchg;

use nalgebra::Matrix;

/// [Matrix representation of a linear system][MRLS].
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix, matrix};
/// use nalgebra_linsys::MatrixReprOfLinSys;
///
/// // x₁ + 2x₂ = 3
/// // 4x₁ + 5x₂ = 6
/// let mut a = MatrixReprOfLinSys::new(matrix![
///    1, 2, 3;
///    4, 5, 6;
/// ]);
///
/// a.row_add(1,0,&-4);
///
/// // x₁ + 2x₂ = 3
/// // -3x₂ = -6
/// assert_eq!(
///   a.0,
///  matrix![
///   1, 2, 3;
///   0, -3, -6;
/// ]);
/// ```
///
/// [MRLS]: http://linear.ups.edu/html/definitions.html
pub struct MatrixReprOfLinSys<T, R, C, S>(pub Matrix<T, R, C, S>);

impl<T, R, C, S> MatrixReprOfLinSys<T, R, C, S> {
    /// Creates a new [matrix representation of a linear system][MRLS] from the given matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::matrix;
    /// use nalgebra_linsys::MatrixReprOfLinSys;
    ///
    /// // x₁ + 2x₂ = 3
    /// // 4x₁ + 5x₂ = 6
    /// let mut a = MatrixReprOfLinSys::new(matrix![
    ///    1, 2, 3;
    ///    4, 5, 6;
    /// ]);
    /// ```
    ///  
    /// [MRLS]: http://linear.ups.edu/html/definitions.html
    pub fn new(matrix: Matrix<T, R, C, S>) -> Self {
        MatrixReprOfLinSys(matrix)
    }

    /// Returns the inner matrix of the given [matrix representation of a linear system][MRLS].
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::matrix;
    /// use nalgebra_linsys::MatrixReprOfLinSys;
    ///
    /// // x₁ + 2x₂ = 3
    /// // 4x₁ + 5x₂ = 6
    /// let mut a = MatrixReprOfLinSys::new(matrix![
    ///   1, 2, 3;
    ///   4, 5, 6;
    /// ]);
    ///
    /// assert_eq!(
    ///  a.0,
    ///  matrix![
    ///    1, 2, 3;
    ///    4, 5, 6;
    /// ]);
    ///
    /// assert_eq!(
    ///   a.to_matrix(),
    ///   matrix![
    ///     1, 2, 3;
    ///     4, 5, 6;
    /// ]);
    /// ```
    /// 
    /// # Notes
    /// 
    /// As opposed to `mrls.0`, a call of this method can, depending on the context,
    /// better convey that the returned (owned) value is a matrix.
    ///
    /// [MRLS]: http://linear.ups.edu/html/definitions.html
    pub fn to_matrix(self) -> Matrix<T, R, C, S> {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::matrix;

    #[test]
    fn row_xchg_works_for_prim_ints() {
        let mut m = MatrixReprOfLinSys::new(matrix!(
            1usize, 2usize;
            3usize, 4usize;
        ));
        unsafe { m.row_xchg_unchecked(row_xchg::PO { row_zbi_1: 0, row_zbi_2: 1 }) };
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
        let mut m = MatrixReprOfLinSys::new(matrix!(
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
    fn row_xchg_works_with_num_rational_entries() {
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // 1/2 3/4
        // 5/6 7/8
        let mut m = MatrixReprOfLinSys::new(matrix!(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(3), BigInt::from(4));
            BigRational::new(BigInt::from(5), BigInt::from(6)),
            BigRational::new(BigInt::from(7), BigInt::from(8));
        ));
        unsafe { m.row_xchg_unchecked(row_xchg::PO { row_zbi_1: 0, row_zbi_2: 1 }) };
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
    fn row_add_works_with_num_rational_entries() {
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // 1/2 3/4
        // 5/6 7/8
        let mut m = MatrixReprOfLinSys::new(matrix!(
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
    fn row_mul_works_with_num_rational_entries() {
        use num_bigint::BigInt;
        use num_rational::BigRational;

        // 1/2 3/4
        // 5/6 7/8
        let mut m = MatrixReprOfLinSys::new(matrix!(
            BigRational::new(BigInt::from(1), BigInt::from(2)),
            BigRational::new(BigInt::from(3), BigInt::from(4));
            BigRational::new(BigInt::from(5), BigInt::from(6)),
            BigRational::new(BigInt::from(7), BigInt::from(8));
        ));
        // 2/1 = 2
        let factor = BigRational::new(BigInt::from(2), BigInt::from(1));
        // Multiplies 0th row by factor
        unsafe { m.row_mul_unchecked(row_mul::PO{ row_zbi: 0, factor: &factor }) };
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
