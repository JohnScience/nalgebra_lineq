//! Module with both safe and unsafe implementations of [elementary row operation] of row addition
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{
    elem_row_op::ElemRowOp, err::BinaryRowIdxOutOfBoundsError, MatrixReprOfLinSys,
};
use core::ops::{AddAssign, Mul};
use nalgebra::{Dim, RawStorageMut, Matrix};

/// The type representing the [elementary row operation] of row addition, i.e. the operation on
/// a matrix where the multiple of one row is added entrywise to another row.
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
///     MatrixReprOfLinSys as  MRLS,
///     elem_row_ops::RowAdd,
/// };
///
/// // x₁ + 2x₂ = 3
/// // 2x₁ = 4
/// let mut m = MRLS::new(matrix![
///    1, 2, 3;
///    2, 0, 4;
/// ]);
///
/// m.perform_elem_row_op(RowAdd {
///    inout_row_zbi: 1,
///    in_row_zbi: 0,
///    factor: &-2
/// }).unwrap();
///
/// // x₁ + 2x₂ = 3
/// // -4x₂ = -2
/// assert_eq!(
///    m.0,
///    matrix![
///      1,  2, 3;
///      0, -4, -2;
/// ]);
/// ```
///
/// [Functionally defined]: https://www.ucfmapper.com/education/various-types-definitions/#:~:text=Functional%20definitions
/// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
/// [parameter objects]: http://principles-wiki.net/patterns:parameter_object
pub struct RowAdd<'a, T> {
    /// The zero-based index of the row to which the scaled second row is added, i.e.
    /// the zero-based index of the "inout row"
    pub inout_row_zbi: usize,
    /// The zero-based index of the row to be scaled and added to the "inout row", i.e.
    /// the zero-based index of the "in row"
    pub in_row_zbi: usize,
    /// The factor by which the "in row" is scaled before summation
    pub factor: &'a T,
}

impl<'a, T, R, C, S> ElemRowOp<Matrix<T,R,C,S>> for RowAdd<'a, T>
where
    T: Clone + AddAssign + Mul<&'a T, Output = T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = BinaryRowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut Matrix<T, R, C, S>) {
        let RowAdd {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor,
        } = self;

        let ncols = m.ncols();

        for j in 0..ncols {
            let corresponding_entry = m[(i_2, j)].to_owned();
            *m.get_unchecked_mut((i_1, j)) += corresponding_entry * factor;
        }
    }

    fn validate(&self, m: &Matrix<T, R, C, S>) -> Result<(), Self::Error> {
        use BinaryRowIdxOutOfBoundsError::*;

        let RowAdd {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor: _unused_factor,
        } = *self;

        let nrows = m.nrows();
        match (i_1, i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            _ => Ok(()),
        }
    }
}

impl<'a, T, R, C, S> ElemRowOp<MatrixReprOfLinSys<T,R,C,S>> for RowAdd<'a, T>
where
    T: Clone + AddAssign + Mul<&'a T, Output = T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = BinaryRowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) {
        self.perform_unchecked(&mut m.0)
    }

    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(), Self::Error> {
        self.validate(&m.0)
    }
}
