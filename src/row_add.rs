//! Module with both safe and unsafe implementations of [elementary row operation] of row addition
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{err::BinaryRowIdxOutOfBoundsError, po::RowAdd as PO, MatrixReprOfLinSys};
use core::ops::{AddAssign, Mul};
use std::borrow::Borrow;
use nalgebra::{Dim, RawStorageMut};

impl<T, R, C, S> MatrixReprOfLinSys<T, R, C, S>
where
    T: Clone + AddAssign,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    /// Sets the row `i_1` to the sum of itself and the scaled by the given `factor` row `i_2`;
    /// without validating row indices.
    ///
    /// # Arguments
    ///
    /// `i_1` - the zero-based index of the row to be modified and whose value is one of the
    /// summands.
    /// `i_2` - the zero-based index of the row to be scaled and whose value after scaling is
    /// the other summand.
    /// `factor` - the factor by which the row `i_2` is scaled before summation.
    ///
    /// # Safety
    ///
    /// The passed values for `i_1` and `i_2` must be actual zero-based indices for the given matrix
    /// (i.e. they can be equal but cannot be greater than or equal to the number of rows).
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::matrix;
    /// use nalgebra_linsys::{
    ///     MatrixReprOfLinSys,
    ///     param_obj::RowAdd as PO,
    /// };
    ///
    /// let mut m = MatrixReprOfLinSys::new(matrix![
    ///    1, 2;
    ///    3, 4;
    /// ]);
    ///
    /// // The call is safe because both 0 and 1 are valid [zero-based] row indices
    /// unsafe { m.row_add_unchecked(PO {
    ///     inout_row_zbi: 1,
    ///     in_row_zbi: 0,
    ///     factor: &-3
    /// })};
    ///
    /// assert_eq!(
    ///    m.0,
    ///    matrix![
    ///      1,  2;
    ///      0, -2;
    /// ]);
    /// ```
    pub unsafe fn row_add_unchecked<'a, 'b>(
        &'a mut self,
        PO {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor,
        }: PO<'b, T>,
    ) where
        T: Mul<&'b T, Output = T> + 'b + AddAssign<T>,
    {
        let ncols = self.0.ncols();
        for j in 0..ncols {
            let corresponding_entry = self.0[(i_2, j)].to_owned();
            *self.0.get_unchecked_mut((i_1, j)) += corresponding_entry * factor.borrow();
        }
    }

    pub fn row_add<'a, 'b>(
        &'a mut self,
        PO {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor,
        }: PO<'b, T>,
    ) -> Result<(), BinaryRowIdxOutOfBoundsError>
    where
        T: Mul<&'b T, Output = T> + 'b + AddAssign<T>,
    {
        use BinaryRowIdxOutOfBoundsError::*;

        let nrows = self.0.nrows();
        match (i_1, i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            #[allow(clippy::unit_arg)]
            _ => Ok(unsafe {
                self.row_add_unchecked(PO {
                    inout_row_zbi: i_1,
                    in_row_zbi: i_2,
                    factor,
                })
            }),
        }
    }
}
