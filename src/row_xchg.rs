//! Module with both safe and unsafe implementations of [elementary row operation] of row exchange
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::po::RowXchg as PO;
use crate::{err::BinaryRowIdxOutOfBoundsError, MatrixReprOfLinSys};
use nalgebra::{Dim, RawStorageMut};

impl<T, R, C, S> MatrixReprOfLinSys<T, R, C, S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    /// Switches rows `i_1` and `i_2` in the matrix without validating row indices.
    ///
    /// # Arguments
    ///
    /// `i_1` and `i_2` - the zero-based indices of the rows to be switched.
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
    ///     param_obj::RowXchg as PO,
    /// };
    ///
    /// let mut m = MatrixReprOfLinSys::new(matrix!(
    ///     1, 2;
    ///     3, 4;
    /// ));
    /// // The call is safe because both 0 and 1 are valid [zero-based] row indices
    /// unsafe { m.row_xchg_unchecked(PO { row_zbi_1: 0, row_zbi_2: 1 }) };
    /// assert_eq!(
    ///     m.0,
    ///     matrix!(
    ///         3, 4;
    ///         1, 2;
    ///     )
    /// );
    /// ```
    ///
    /// # Notes
    ///
    /// Unlike [`nalgebra::base::Matrix::swap_rows`], this method doesn't require the entries
    /// to implement [`nalgebra::base::Scalar`] and doesn't perform bounds checking.
    ///
    /// The safe counterpart of this method is [`MatrixReprOfLinSys::row_xchg`].
    ///
    /// The remaining two unsafe implementations of elementary row operations are
    /// [`MatrixReprOfLinSys::row_add_unchecked`] and [`MatrixReprOfLinSys::row_mul_unchecked`].
    pub unsafe fn row_xchg_unchecked(
        &mut self,
        PO {
            row_zbi_1: i_1,
            row_zbi_2: i_2,
        }: PO,
    ) {
        let ncols = self.0.ncols();
        (0..ncols)
            .map(|j| ((i_1, j), (i_2, j)))
            .for_each(|(row_col1, row_col2)| {
                self.0.swap_unchecked(row_col1, row_col2);
            });
    }

    /// Attempts to switch rows `i_1` and `i_2` in the matrix after performing bounds checking.
    ///
    /// # Arguments
    ///
    /// `i_1` and `i_2` - the zero-based indices of the rows to be switched.
    ///
    /// # Returns
    ///
    /// [`Result`]`<(), `[`BinaryRowIdxOutOfBoundsError`]`>` - [`Result::Ok`] if the rows were successfully
    /// switched and [`Result::Err`] otherwise, in case if at least one of the indices is out of bounds.
    ///
    /// # Notes
    ///
    /// Unlike [`nalgebra::base::Matrix::swap_rows`], this method doesn't require the entries
    /// to implement [`nalgebra::base::Scalar`] and doesn't perform bounds checking.
    ///
    /// The unsafe version of this method is [`MatrixReprOfLinSys::row_xchg_unchecked`].
    ///
    /// The remaining two safe implementations of elementary row operations are
    /// [`MatrixReprOfLinSys::row_add`] and [`MatrixReprOfLinSys::row_mul`].
    pub fn row_xchg(
        &mut self,
        PO {
            row_zbi_1: i_1,
            row_zbi_2: i_2,
        }: PO,
    ) -> Result<(), BinaryRowIdxOutOfBoundsError> {
        use BinaryRowIdxOutOfBoundsError::*;

        let nrows = self.0.nrows();
        match (i_1, i_2) {
            (i_1, i_2) if i_1 >= nrows && i_2 >= nrows => Err(BothIdcesOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_1 >= nrows => Err(FirstIdxOutOfBounds((i_1, i_2))),
            (i_1, i_2) if i_2 >= nrows => Err(SecondIdxOutOfBounds((i_1, i_2))),
            #[allow(clippy::unit_arg)]
            _ => Ok(unsafe {
                self.row_xchg_unchecked(PO {
                    row_zbi_1: i_1,
                    row_zbi_2: i_2,
                })
            }),
        }
    }
}
