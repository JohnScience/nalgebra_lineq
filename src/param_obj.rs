//! Module with [parameter objects] for both safe and unsafe implementations of [elementary row operations]
//! 
//! [elementary row operations]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
//! [parameter objects]: https://en.wikipedia.org/wiki/Parameter_object

use crate::MatrixReprOfLinSys;

pub trait ElementaryRowOperation<T,R,C,S>: Sized {
    type Error;
    unsafe fn perform_unchecked(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) -> ();
    fn validate(&self, m: &MatrixReprOfLinSys<T,R,C,S>) -> Result<(),Self::Error>;
    fn perform(self, m: &mut MatrixReprOfLinSys<T,R,C,S>) -> Result<(),Self::Error> {
        self.validate(m)?;
        unsafe { self.perform_unchecked(m) };
        Ok(())
    }
}

/// [Parameter object] for [`MatrixReprOfLinSys::row_xchg`][`crate::MatrixReprOfLinSys::row_xchg`]
/// and [`MatrixReprOfLinSys::row_xchg_unchecked`][`crate::MatrixReprOfLinSys::row_xchg_unchecked`].
///
/// # Example
///
/// ```
/// use nalgebra::matrix;
/// use nalgebra_linsys::{
///    param_obj::RowXchg as PO,
///    MatrixReprOfLinSys as MRLS,
/// };
///
/// // x₁ + 2x₂ = 3
/// // 4x₁ + 5x₂ = 6
/// let mut m = MRLS::new(matrix![
///   1, 2, 3;
///   4, 5, 6;
/// ]);
///
/// unsafe { m.row_xchg_unchecked(PO {
///    row_zbi_1: 1,
///    row_zbi_2: 0,
/// })};
///
/// assert_eq!(
///  m.0,
///  matrix![
///    4, 5, 6;
///    1, 2, 3;
/// ]);
/// ```
///
/// [Parameter object]: http://principles-wiki.net/patterns:parameter_object
pub struct RowXchg {
    /// The zero-based index of the first row to be exchanged
    pub row_zbi_1: usize,
    /// The zero-based index of the second row to be exchanged
    pub row_zbi_2: usize,
}

/// [Parameter object] for [`MatrixReprOfLinSys::row_add`][`crate::MatrixReprOfLinSys::row_add`] and
/// [`MatrixReprOfLinSys::row_add_unchecked`][`crate::MatrixReprOfLinSys::row_add_unchecked`].
///
/// # Generic arguments
///
/// `'a` - the lifetime of the `factor`;
///
/// `T` - the type of the factor.
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
///
/// [Parameter object]: http://principles-wiki.net/patterns:parameter_object
pub struct RowAdd<'a,T>
{
    /// The zero-based index of the row to which the scaled second row is added, i.e.
    /// the zero-based index of the inout row
    pub inout_row_zbi: usize,
    /// The zero-based index of the row to be scaled and added to the inout row, i.e.
    /// the zero-based index of the in row
    pub in_row_zbi: usize,
    /// The factor by which the in row is scaled before summation
    pub factor: &'a T,
}

/// [Parameter object] for [`MatrixReprOfLinSys::row_mul`][`crate::MatrixReprOfLinSys::row_mul`] and
/// [`MatrixReprOfLinSys::row_mul_unchecked`][`crate::MatrixReprOfLinSys::row_mul_unchecked`].
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
///     param_obj::RowMul as PO,
/// };
///
/// let mut m = MatrixReprOfLinSys::new(matrix![
///    1, 2;
///    3, 4;
/// ]);
///
/// // The call is safe because both 0 and 1 are valid [zero-based] row indices
/// unsafe { m.row_mul_unchecked(PO {
///     row_zbi: 0,
///     factor: &2
/// })};
///
/// assert_eq!(
///    m.0,
///    matrix![
///      2,  4;
///      3, 4;
/// ]);
/// ```
///
/// [Parameter object]: http://principles-wiki.net/patterns:parameter_object
pub struct RowMul<'a, T> {
    /// The zero-based index of the row to be scaled
    pub row_zbi: usize,
    /// The factor by which the row is scaled
    pub factor: &'a T,
}
