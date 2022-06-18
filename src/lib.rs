#[cfg(any(test, doctest))]
extern crate num_bigint;
#[cfg(any(test, doctest))]
extern crate num_rational;

pub mod err;
pub mod param_obj;

// implementations
mod row_add;
mod row_mul;
mod row_xchg;

// convenience alias
pub(crate) use param_obj as po;

use nalgebra::Matrix;
use po::ElementaryRowOperation;

/// [Matrix representation of a linear system][MRLS].
///
/// # Example
///
/// ```
/// use nalgebra::{Matrix, matrix};
/// use nalgebra_linsys::{
///     MatrixReprOfLinSys,
///     param_obj as po,
/// };
///
/// // x₁ + 2x₂ = 3
/// // 4x₁ + 5x₂ = 6
/// let mut a = MatrixReprOfLinSys::new(matrix![
///    1, 2, 3;
///    4, 5, 6;
/// ]);
///
/// a.row_add(po::RowAdd {
///     // the zero-based index of the row to which the scaled second row is added, i.e.
///     // the zero-based index of the inout row;
///     inout_row_zbi: 1,
///     // the zero-based index of the row whose scaled value is added to the inout row,
///     // i.e. the zero-based index of the in row;
///     in_row_zbi: 0,
///     // the factor by which the in row is scaled before summation.
///     factor: &-4
/// });
///
/// // x₁ + 2x₂ = 3
/// // -3x₂ = -6
/// assert_eq!(
///   a.0,
///  matrix![
///   1, 2, 3;
///   0, -3, -6;
/// ]);
/// ```
///
/// [MRLS]: http://linear.ups.edu/html/definitions.html
pub struct MatrixReprOfLinSys<T, R, C, S>(pub Matrix<T, R, C, S>);

impl<T, R, C, S> MatrixReprOfLinSys<T, R, C, S> {
    /// Creates a new [matrix representation of a linear system][MRLS] from the given matrix.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::matrix;
    /// use nalgebra_linsys::{MatrixReprOfLinSys as MRLS};
    ///
    /// // x₁ + 2x₂ = 3
    /// // 4x₁ + 5x₂ = 6
    /// let mut a = MRLS::new(matrix![
    ///    1, 2, 3;
    ///    4, 5, 6;
    /// ]);
    /// ```
    ///  
    /// [MRLS]: http://linear.ups.edu/html/definitions.html
    pub fn new(matrix: Matrix<T, R, C, S>) -> Self {
        MatrixReprOfLinSys(matrix)
    }

    /// Turns the given [matrix representation of a linear system][MRLS] into 
    /// an object of [nalgebra::Matrix] type.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::matrix;
    /// use nalgebra_linsys::{MatrixReprOfLinSys as MRLS};
    ///
    /// // x₁ + 2x₂ = 3
    /// // 4x₁ + 5x₂ = 6
    /// let mut a = MRLS::new(matrix![
    ///   1, 2, 3;
    ///   4, 5, 6;
    /// ]);
    ///
    /// assert_eq!(
    ///  a.0,
    ///  matrix![
    ///    1, 2, 3;
    ///    4, 5, 6;
    /// ]);
    ///
    /// assert_eq!(
    ///   a.to_matrix(),
    ///   matrix![
    ///     1, 2, 3;
    ///     4, 5, 6;
    /// ]);
    /// ```
    ///
    /// # Notes
    ///
    /// As opposed to `m.0`, a call of this method can, depending on the context,
    /// better convey that the returned (owned) value is a matrix.
    ///
    /// [MRLS]: http://linear.ups.edu/html/definitions.html
    pub fn to_matrix(self) -> Matrix<T, R, C, S> {
        self.0
    }
}

impl<T,R,C,S> MatrixReprOfLinSys<T,R,C,S> {
    pub fn perform_elementary_row_operation<O>(&mut self, o: O) -> Result<(), O::Error>
    where
        O: ElementaryRowOperation<T,R,C,S>
    {
        o.perform(self)
    }
}
