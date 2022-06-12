use nalgebra::{Matrix, Dim, RawStorageMut, RawStorage};
use core::ops::{AddAssign, Mul, Index, IndexMut};

pub struct MatrixReprOfLinEq<T,R,C,S>(pub Matrix<T,R,C,S>);

/// impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S> {
///     pub fn new(matrix: Matrix<T,R,C,S>) -> Self {
///         MatrixReprOfLinEq(matrix)
///     }
/// 
///     pub fn to_matrix(self) -> Matrix<T,R,C,S> {
///         self.0
///     }
/// }

impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
where
    R: Dim,
    C: Dim,
    S: RawStorageMut<T,R,C>
{
    /// Switches rows `i_1` and `i_2` in the matrix.
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it does not check if the indices are valid.
    unsafe fn row_xchg(&mut self, i_1: usize, i_2: usize)
    {
        let ncols = self.0.ncols();
        (0..ncols)
            .map(|j| ((i_1, j), (i_2, j)))
            .for_each(|(row_col1, row_col2)| {
                self.0.swap_unchecked(row_col1, row_col2);
            }
        );
    }
}

// impl<T,R,C,S> MatrixReprOfLinEq<T,R,C,S>
// where
//     T: Mul<Output=T> + Clone,
//     R: Dim,
//     C: Dim,
//     S: RawStorage<T,R,C> + RawStorageMut<T,R,C>,
// {
//     unsafe fn row_add<'a>(&'a mut self, i_1: usize, i_2: usize, factor: T)
//     where
//         &'a mut T: AddAssign<T>,
//     {
//         let ncols = self.0.ncols();
//         for j in 0..ncols {
//             let entry = (*self.0.data.get_address_unchecked_mut(i_1, j)).clone();
//             let prod = factor.clone() * entry;
//             self.0.data.get_unchecked_mut(i_1, j).add_assign(prod);
//         }
//     }
// 
//     unsafe fn row_mul(&mut self, i: usize, factor: T) {
//         let ncols = self.0.ncols();
//         for j in 0..ncols {
//             self.0[(i, j)] = self.0[(i, j)] * factor;
//         }
//     }
// }
// 
// #[cfg(test)]
// mod tests {
//     use super::*;
//     use nalgebra::{Matrix, Dim, matrix};
// 
//     #[test]
//     fn it_works() {
//         let m = MatrixReprOfLinEq::new(matrix!(
//             1usize, 2usize,
//             3usize, 4usize,
//         ));
//         unsafe { m.row_xchg(0, 1) };
//         assert_eq!(
//             m.0,
//             matrix!(
//                 3usize, 4usize,
//                 1usize, 2usize
//             )
//         );
//     }
// }