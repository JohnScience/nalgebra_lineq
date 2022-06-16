//! Module with error types for safe implementations of [elementary row operations]
//!
//! [elementary row operations]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
use thiserror::Error;

/// Out-of-bounds error type for [`MatrixReprOfLinSys::row_xchg`] and [`MatrixReprOfLinSys::row_add`].
///
/// # Notes
///
/// As opposed to [`RowIdxOutOfBoundsError`], this error can be caused by incorrect value(-s) of at most
/// **two** row indices.
///
/// [`MatrixReprOfLinSys::row_xchg`]: [`crate::MatrixReprOfLinSys::row_xchg`]
/// [`MatrixReprOfLinSys::row_add`]: [`crate::MatrixReprOfLinSys::row_add`]
#[derive(Error, Debug)]
pub enum BinaryRowIdxOutOfBoundsError {
    #[error("First row index is out of bounds: {0:?}")]
    FirstIdxOutOfBounds((usize, usize)),
    #[error("Second row index is out of bounds: {0:?}")]
    SecondIdxOutOfBounds((usize, usize)),
    #[error("Both row indices are out of bounds: {0:?}")]
    BothIdcesOutOfBounds((usize, usize)),
}

/// Out-of-bounds error type for [`MatrixReprOfLinSys::row_mul`].
///
/// # Notes
///
/// As opposed to [`BinaryRowIdxOutOfBoundsError`], this error is caused by the incorrect value of
/// only one row index instead of two, so there's no room for doubt which index has incorrect value
/// (or whether they both have incorrect values).
///
/// [`MatrixReprOfLinSys::row_mul`]: [`crate::MatrixReprOfLinSys::row_mul`]
#[derive(Error, Debug)]
#[error("Row index is out of bounds: {0:?}")]
pub struct RowIdxOutOfBoundsError(pub(crate) usize);
