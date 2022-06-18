//! Module with both safe and unsafe implementations of [elementary row operation] of row multiplication
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{err::RowIdxOutOfBoundsError, po::{RowMul as PO, ElementaryRowOperation}, MatrixReprOfLinSys};
use core::ops::MulAssign;
use nalgebra::{Dim, RawStorageMut};

impl<'a,T,R,C,S> ElementaryRowOperation<T,R,C,S> for PO<'a, T>
where
    T: Clone + MulAssign<&'a T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>
{
    type Error = RowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) -> () {
        let PO { row_zbi: i, factor } = self;

        let ncols = m.0.ncols();
        for j in 0..ncols {
            *m.0.get_unchecked_mut((i, j)) *= factor;
        }
    }

    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(),Self::Error> {
        let row_zero_based_idx = self.row_zbi;
        let nrows = m.0.nrows();

        if row_zero_based_idx >= nrows { Err(RowIdxOutOfBoundsError(row_zero_based_idx)) }
        else { Ok(()) }
    }
}
