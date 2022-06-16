//! Module with both safe and unsafe implementations of [elementary row operation] of row multiplication
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{err::RowIdxOutOfBoundsError, po::RowMul as PO, MatrixReprOfLinSys};
use core::ops::MulAssign;
use nalgebra::{Dim, RawStorageMut};

impl<T, R, C, S> MatrixReprOfLinSys<T, R, C, S>
where
    T: Clone + MulAssign,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    /// Multiplies a row by a scalar.
    ///
    /// # Safety
    ///
    /// This function is unsafe because it does not check if the row index is valid.
    pub unsafe fn row_mul_unchecked<'a>(&mut self, PO { row_zbi: i, factor }: PO<'a, T>)
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
            #[allow(clippy::unit_arg)]
            Ok(unsafe { self.row_mul_unchecked(PO { row_zbi: i, factor }) })
        }
    }
}
