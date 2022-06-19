[![crates.io](https://img.shields.io/crates/v/nalgebra_linsys.svg)][`nalgebra_linsys`]
[![crates.io](https://img.shields.io/crates/d/nalgebra_linsys.svg)][`nalgebra_linsys`]

# Solving [linear system]s with [`nalgebra`] using [elementary row operations][ero]

This is a crate that you might want to use if you're fine with **suboptimal performance** and, for example, want to have a library that would offer correct yet not necessarily optimized implementations of [elementary row operations], [Gaussian], and/or [Gauss-Jordan elimination].

# Notes

At the time of writing, [Gaussian], and/or [Gauss-Jordan elimination] are not provided.

[`nalgebra_linsys`]: https://crates.io/crates/nalgebra_linsys
[linear system]: https://en.wikipedia.org/wiki/System_of_linear_equations
[`nalgebra`]: https://crates.io/crates/nalgebra
[ero]: https://www.math.ucdavis.edu/~linear/old/notes3.pdf
[Gaussian]: https://en.wikipedia.org/wiki/Row_echelon_form
[Gauss-Jordan elimination]: https://online.stat.psu.edu/statprogram/reviews/matrix-algebra/gauss-jordan-elimination
[parameter objects]: http://principles-wiki.net/patterns:parameter_object
[Zero-based numbering]: https://en.wikipedia.org/wiki/Zero-based_numbering