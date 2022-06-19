//! Module with [parameter objects] for both safe and unsafe implementations of [elementary row operations]
//!
//! [elementary row operations]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
//! [parameter objects]: https://en.wikipedia.org/wiki/Parameter_object

/// The trait whose implementors represent [elementary row operations][ero] acting on a given structure
/// (for example, on a matrix).
/// 
/// [Functionally defined], its implementers are [parameter objects] for
/// [`MatrixReprOfLinSys::perform_elem_row_op`][`crate::MatrixReprOfLinSys::perform_elem_row_op`].
/// 
/// # Generic arguments
/// 
/// `T` - the type on which the [elementary row operations][ero] act.
/// 
/// # Notes
/// 
/// The library provides implementations of the trait parameterization over T =
/// [`MatrixReprOfLinSys`][`crate::MatrixReprOfLinSys`]. You may want to implement it
/// for your own type if you want to, for example, output intermediate results in the chain of
/// transformations.
/// 
/// [ero]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
/// [parameter objects]: http://principles-wiki.net/patterns:parameter_object
/// [Functionally defined]: https://www.ucfmapper.com/education/various-types-definitions/#:~:text=Functional%20definitions
pub trait ElemRowOp<T>: Sized {
    type Error;
    /// Performs the [elementary row operation] on the given structure without validation of the
    /// internal state describing the operation, such as bounds checking of the indices of rows.
    /// 
    /// # Arguments
    /// 
    /// `m` - the matrix or any other structure on which the [elementary row operation] is to be
    /// performed.
    /// 
    /// # Safety
    /// 
    /// [`ElemRowOp::validate`] must be executed successfully before performing the operation.
    /// 
    /// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
    unsafe fn perform_unchecked(self, m: &mut T);
    /// Checks whether the internal state describing the [elementary row operation] is valid.
    /// For example, the validation may include bounds checking of the row indices.
    /// 
    /// # Arguments
    /// 
    /// `m` - the matrix or any other structure on which the [elementary row operation] is to be
    /// performed.
    /// 
    /// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
    fn validate(&self, m: &T) -> Result<(), Self::Error>;
    /// Performs the [elementary row operation] on the given structure.
    /// 
    /// [elementary row operation]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
    fn perform(self, m: &mut T) -> Result<(), Self::Error> {
        self.validate(m)?;
        unsafe { self.perform_unchecked(m) };
        Ok(())
    }
}
