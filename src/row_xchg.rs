//! Module with both safe and unsafe implementations of [elementary row operation] of row exchange
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{
    elem_row_op::ElemRowOp, err::BinaryRowIdxOutOfBoundsError, MatrixReprOfLinSys,
};
use nalgebra::{Dim, RawStorageMut};

/// The type representing the [elementary row operation] of row exchange, i.e. the operation
/// on a matrix that swaps entries in two of its rows.
/// 
/// [Functionally defined], it is one of possible [parameter objects] for
/// [`MatrixReprOfLinSys::perform_elem_row_op`][`crate::MatrixReprOfLinSys::perform_elem_row_op`].
///
/// # Example
///
/// ```
/// use nalgebra::matrix;
/// use nalgebra_linsys::{
///    MatrixReprOfLinSys as MRLS,
///    elem_row_ops::RowXchg,
/// };
///
/// // x₁ + 2x₂ = 3
/// // 4x₁ + 5x₂ = 6
/// let mut m = MRLS::new(matrix![
///   1, 2, 3;
///   4, 5, 6;
/// ]);
///
/// m.perform_elem_row_op(RowXchg {
///     row_zbi_1: 1,
///     row_zbi_2: 0,
/// }).unwrap();
/// 
/// // 4x₁ + 5x₂ = 6
/// // x₁ + 2x₂ = 3
/// assert_eq!(
///  m.0,
///  matrix![
///    4, 5, 6;
///    1, 2, 3;
/// ]);
/// ```
/// 
/// [Functionally defined]: https://www.ucfmapper.com/education/various-types-definitions/#:~:text=Functional%20definitions
/// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
/// [parameter objects]: http://principles-wiki.net/patterns:parameter_object
pub struct RowXchg {
    /// The zero-based index of the first row to be exchanged
    pub row_zbi_1: usize,
    /// The zero-based index of the second row to be exchanged
    pub row_zbi_2: usize,
}

impl<T, R, C, S> ElemRowOp<MatrixReprOfLinSys<T,R,C,S>> for RowXchg
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = BinaryRowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T, R, C, S>) {
        let RowXchg {
            row_zbi_1: i_1,
            row_zbi_2: i_2,
        } = self;

        let ncols = m.0.ncols();
        (0..ncols)
            .map(|j| ((i_1, j), (i_2, j)))
            .for_each(|(row_col1, row_col2)| {
                m.0.swap_unchecked(row_col1, row_col2);
            });
    }

    fn validate(&self, m: &MatrixReprOfLinSys<T, R, C, S>) -> Result<(), Self::Error> {
        use BinaryRowIdxOutOfBoundsError::*;

        let RowXchg {
            row_zbi_1: i_1,
            row_zbi_2: i_2,
        } = *self;

        let nrows = m.0.nrows();

        match (i_1, i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            _ => Ok(()),
        }
    }
}
