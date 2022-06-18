//! Module with both safe and unsafe implementations of [elementary row operation] of row exchange
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{
    err::BinaryRowIdxOutOfBoundsError,
    MatrixReprOfLinSys,
    po::{RowXchg as PO, ElementaryRowOperation},
};
use nalgebra::{Dim, RawStorageMut};

impl<T,R,C,S> ElementaryRowOperation<T,R,C,S> for PO
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>
{
    type Error = BinaryRowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) -> () {
        let PO {
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

    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(),Self::Error> {
        use BinaryRowIdxOutOfBoundsError::*;
        
        let PO {
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
