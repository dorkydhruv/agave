use thiserror::Error;

#[derive(Error, Debug)]
pub enum DenseError {
    #[error("Invalid matrix dimensions")]
    InvalidDimensions,
    #[error("Index out of bounds")]
    IndexOutOfBounds,
    #[error("Matrix is singular")]
    SingularMatrix,
}

/// Dense matrix over GF(2) - exact port of mod2dense
#[derive(Debug, Clone)]
pub struct Mod2Dense {
    n_rows: usize,
    n_cols: usize,
    /// Data stored as bits packed into u32s
    /// Each row is stored in consecutive u32s
    data: Vec<u32>,
    /// Number of u32s needed per row
    words_per_row: usize,
}

impl Mod2Dense {
    /// ALLOCATE SPACE FOR A DENSE MOD2 MATRIX - exact port of mod2dense_allocate
    pub fn allocate(n_rows: usize, n_cols: usize) -> Result<Self, DenseError> {
        if n_rows == 0 || n_cols == 0 {
            return Err(DenseError::InvalidDimensions);
        }

        let words_per_row = (n_cols + 31) / 32; // Ceiling division
        let data = vec![0u32; n_rows * words_per_row];

        Ok(Self {
            n_rows,
            n_cols,
            data,
            words_per_row,
        })
    }

    pub fn rows(&self) -> usize {
        self.n_rows
    }
    pub fn cols(&self) -> usize {
        self.n_cols
    }

    /// GET AN ELEMENT OF A DENSE MOD2 MATRIX - exact port of mod2dense_get
    pub fn get(&self, row: usize, col: usize) -> bool {
        if row >= self.n_rows || col >= self.n_cols {
            return false;
        }

        let word_index = row * self.words_per_row + col / 32;
        let bit_index = col % 32;

        (self.data[word_index] & (1u32 << bit_index)) != 0
    }

    /// SET AN ELEMENT OF A DENSE MOD2 MATRIX - exact port of mod2dense_set
    pub fn set(&mut self, row: usize, col: usize, value: bool) -> Result<(), DenseError> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(DenseError::IndexOutOfBounds);
        }

        let word_index = row * self.words_per_row + col / 32;
        let bit_index = col % 32;
        let mask = 1u32 << bit_index;

        if value {
            self.data[word_index] |= mask;
        } else {
            self.data[word_index] &= !mask;
        }

        Ok(())
    }

    /// FLIP AN ELEMENT OF A DENSE MOD2 MATRIX - exact port of mod2dense_flip
    pub fn flip(&mut self, row: usize, col: usize) -> Result<(), DenseError> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(DenseError::IndexOutOfBounds);
        }

        let word_index = row * self.words_per_row + col / 32;
        let bit_index = col % 32;
        let mask = 1u32 << bit_index;

        self.data[word_index] ^= mask;
        Ok(())
    }

    /// CLEAR A DENSE MATRIX TO ALL ZEROS - exact port of mod2dense_clear
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// COPY A DENSE MATRIX - exact port of mod2dense_copy
    pub fn copy_from(&mut self, source: &Mod2Dense) -> Result<(), DenseError> {
        if self.n_rows != source.n_rows || self.n_cols != source.n_cols {
            return Err(DenseError::InvalidDimensions);
        }

        self.data.copy_from_slice(&source.data);
        Ok(())
    }

    /// GAUSSIAN ELIMINATION FOR INVERSION - exact port of mod2dense_invert
    pub fn invert(&mut self) -> Result<(), DenseError> {
        if self.n_rows != self.n_cols {
            return Err(DenseError::InvalidDimensions);
        }

        let n = self.n_rows;

        // Forward elimination
        for i in 0..n {
            // Find pivot
            let mut pivot_row = None;
            for j in i..n {
                if self.get(j, i) {
                    pivot_row = Some(j);
                    break;
                }
            }

            let pivot_row = pivot_row.ok_or(DenseError::SingularMatrix)?;

            // Swap rows if needed
            if pivot_row != i {
                self.swap_rows(i, pivot_row);
            }

            // Eliminate column
            for j in 0..n {
                if j != i && self.get(j, i) {
                    self.add_row(j, i);
                }
            }
        }

        Ok(())
    }

    /// MULTIPLY TWO DENSE MOD2 MATRICES - exact port of mod2dense_multiply
    pub fn multiply(a: &Mod2Dense, b: &Mod2Dense) -> Result<Mod2Dense, DenseError> {
        if a.n_cols != b.n_rows {
            return Err(DenseError::InvalidDimensions);
        }

        let mut result = Mod2Dense::allocate(a.n_rows, b.n_cols)?;

        for i in 0..a.n_rows {
            for j in 0..b.n_cols {
                let mut sum = false;
                for k in 0..a.n_cols {
                    if a.get(i, k) && b.get(k, j) {
                        sum = !sum; // XOR in GF(2)
                    }
                }
                if sum {
                    result.set(i, j, true)?;
                }
            }
        }

        Ok(result)
    }

    /// Swap two rows
    fn swap_rows(&mut self, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        for word_offset in 0..self.words_per_row {
            let idx1 = row1 * self.words_per_row + word_offset;
            let idx2 = row2 * self.words_per_row + word_offset;
            self.data.swap(idx1, idx2);
        }
    }

    /// Add row2 to row1 (XOR in GF(2))
    fn add_row(&mut self, row1: usize, row2: usize) {
        for word_offset in 0..self.words_per_row {
            let idx1 = row1 * self.words_per_row + word_offset;
            let idx2 = row2 * self.words_per_row + word_offset;
            self.data[idx1] ^= self.data[idx2];
        }
    }

    /// Get a row as a bit vector
    pub fn get_row(&self, row: usize) -> Vec<bool> {
        let mut result = Vec::with_capacity(self.n_cols);
        for col in 0..self.n_cols {
            result.push(self.get(row, col));
        }
        result
    }

    /// Set a row from a bit vector
    pub fn set_row(&mut self, row: usize, bits: &[bool]) -> Result<(), DenseError> {
        if bits.len() != self.n_cols {
            return Err(DenseError::InvalidDimensions);
        }

        for (col, &bit) in bits.iter().enumerate() {
            self.set(row, col, bit)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matrix_basic() {
        let mut matrix = Mod2Dense::allocate(3, 4).unwrap();

        // Test set and get
        matrix.set(1, 2, true).unwrap();
        assert!(matrix.get(1, 2));
        assert!(!matrix.get(1, 1));

        // Test flip
        matrix.flip(1, 2).unwrap();
        assert!(!matrix.get(1, 2));

        matrix.flip(0, 0).unwrap();
        assert!(matrix.get(0, 0));
    }

    #[test]
    fn test_dense_matrix_multiply() {
        let mut a = Mod2Dense::allocate(2, 3).unwrap();
        let mut b = Mod2Dense::allocate(3, 2).unwrap();

        // Set up test matrices
        a.set(0, 0, true).unwrap();
        a.set(0, 2, true).unwrap();
        a.set(1, 1, true).unwrap();

        b.set(0, 1, true).unwrap();
        b.set(1, 0, true).unwrap();
        b.set(2, 1, true).unwrap();

        let result = Mod2Dense::multiply(&a, &b).unwrap();

        // Check results
        assert!(!result.get(0, 0)); // 1*0 + 0*1 + 1*0 = 0
        assert!(!result.get(0, 1)); // 1*1 + 0*0 + 1*1 = 0 (XOR)
        assert!(result.get(1, 0)); // 0*0 + 1*1 + 0*0 = 1
        assert!(!result.get(1, 1)); // 0*1 + 1*0 + 0*1 = 0
    }
}
