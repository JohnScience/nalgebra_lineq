//! Module with both safe and unsafe implementations of [elementary row operation] of row addition
//!
//! [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf

use crate::{err::BinaryRowIdxOutOfBoundsError, po::{RowAdd as PO, ElementaryRowOperation}, MatrixReprOfLinSys};
use core::ops::{AddAssign, Mul};
use nalgebra::{Dim, RawStorageMut};

impl<'a,T,R,C,S> ElementaryRowOperation<T,R,C,S> for PO<'a,T>
where
    T: Clone + AddAssign + Mul<&'a T,Output=T>,
    R: Dim,
    C: Dim,
    S: RawStorageMut<T, R, C>,
{
    type Error = BinaryRowIdxOutOfBoundsError;

    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) -> () {
        let PO {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor,
        } = self;
        
        let ncols = m.0.ncols();

        for j in 0..ncols {
            let corresponding_entry = m.0[(i_2, j)].to_owned();
            *m.0.get_unchecked_mut((i_1, j)) += corresponding_entry * factor;
        }
    }

    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(),Self::Error> {
        use BinaryRowIdxOutOfBoundsError::*;

        let PO {
            inout_row_zbi: i_1,
            in_row_zbi: i_2,
            factor: _unused_factor,
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
