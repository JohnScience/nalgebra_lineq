//! Module with both safe and unsafe implementations of [elementary row operation] of row multiplication
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{elem_row_op::ElemRowOp, err::RowIdxOutOfBoundsError, MatrixReprOfLinSys};
use core::ops::MulAssign;
use nalgebra::{Dim, RawStorageMut, Matrix};

/// The type representing the [elementary row operation] of row multiplication, i.e. the operation on
/// a matrix where one row is scaled by the same factor in every entry.
/// 
/// [Functionally defined], it is one of possible [parameter objects] for
/// [`MatrixReprOfLinSys::perform_elem_row_op`][`crate::MatrixReprOfLinSys::perform_elem_row_op`].
///
/// # Generic arguments
///
/// `'a` - the lifetime of the `factor`;
///
/// `T` - the type of the `factor`.
///
/// # Example
///
/// ```
/// use nalgebra::matrix;
/// use nalgebra_linsys::{
///     MatrixReprOfLinSys,
///     elem_row_ops::RowMul,
/// };
/// // x₁ + 2x₂ = 3
/// // 2x₁ + 4x₂ = 5
/// let mut m = MatrixReprOfLinSys::new(matrix![
///    1, 2, 3;
///    2, 4, 5;
/// ]);
///
/// m.perform_elem_row_op(RowMul {
///   row_zbi: 0,
///   factor: &2
/// }).unwrap();
///
/// // 2x₁ + 4x₂ = 6
/// // 2x₁ + 4x₂ = 5
/// assert_eq!(
///    m.0,
///    matrix![
///      2, 4, 6;
///      2, 4, 5;
/// ]);
/// ```
///
/// [Parameter object]: http://principles-wiki.net/patterns:parameter_object
/// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
/// [Functionally defined]: https://www.ucfmapper.com/education/various-types-definitions/#:~:text=Functional%20definitions
pub struct RowMul<'a, T> {
    /// The zero-based index of the row to be scaled
    pub row_zbi: usize,
    /// The factor by which the row is scaled
    pub factor: &'a T,
}

impl<'a, T, R, C, S> ElemRowOp<Matrix<T,R,C,S>> for RowMul<'a, T>
where
    T: Clone + MulAssign<&'a T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = RowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut Matrix<T, R, C, S>) {
        let RowMul { row_zbi: i, factor } = self;

        let ncols = m.ncols();
        for j in 0..ncols {
            *m.get_unchecked_mut((i, j)) *= factor;
        }
    }

    fn validate(&self, m: &Matrix<T, R, C, S>) -> Result<(), Self::Error> {
        let row_zero_based_idx = self.row_zbi;
        let nrows = m.nrows();

        if row_zero_based_idx >= nrows {
            Err(RowIdxOutOfBoundsError(row_zero_based_idx))
        } else {
            Ok(())
        }
    }
}

impl<'a, T, R, C, S> ElemRowOp<MatrixReprOfLinSys<T,R,C,S>> for RowMul<'a, T>
where
    T: Clone + MulAssign<&'a T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = RowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) {
        self.perform_unchecked(&mut m.0)
    }

    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(), Self::Error> {
        self.validate(&m.0)
    }
}
