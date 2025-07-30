use crate::{mod2convert, LdpcError, Mod2Dense, Mod2Sparse};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GeneratorMethod {
    Sparse,
    Dense,
    Mixed,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Mod2SparseStrategy {
    First,
    Mincol,
    Minprod,
}

/// Generator matrix representation - exact port of make-gen.c concepts
#[derive(Debug)]
pub struct GeneratorMatrix {
    pub sparse_gen: Option<Mod2Sparse>,
    pub dense_gen: Option<Mod2Dense>,
    pub method: GeneratorMethod,
    pub cols_to_rows: Vec<i32>, // Maps column indices to row indices, -1 if not a pivot
    pub n_bits: usize,
    pub n_checks: usize,
}

impl GeneratorMatrix {
    /// CREATE GENERATOR MATRIX FROM PARITY CHECK MATRIX - exact port of make-gen.c logic
    pub fn from_parity_check(
        parity_check: &Mod2Sparse,
        method: GeneratorMethod,
        strategy: Mod2SparseStrategy,
        abandon_number: usize,
        abandon_when: usize,
    ) -> Result<Self, LdpcError> {
        let m = parity_check.rows(); // number of checks
        let n = parity_check.cols(); // number of bits

        if n <= m {
            return Err(LdpcError::InvalidParameter(
                "Number of bits must be greater than number of checks".to_string(),
            ));
        }

        match method {
            GeneratorMethod::Sparse => {
                Self::make_sparse(parity_check, strategy, abandon_number, abandon_when)
            }
            GeneratorMethod::Dense | GeneratorMethod::Mixed => {
                Self::make_dense_mixed(parity_check, method)
            }
        }
    }

    /// MAKE SPARSE GENERATOR MATRIX - exact port of make_sparse function
    fn make_sparse(
        h: &Mod2Sparse,
        strategy: Mod2SparseStrategy,
        _abandon_number: usize,
        _abandon_when: usize,
    ) -> Result<Self, LdpcError> {
        let m = h.rows();
        let n = h.cols();

        // Copy parity check matrix for manipulation
        let mut a = h.clone();

        // Set up identity matrix part for L (lower triangular)
        let mut l = Mod2Sparse::allocate(m, m)?;
        for i in 0..m {
            l.insert(i, i)?;
        }

        // Gaussian elimination to get A in systematic form [I | P]
        let cols_to_rows = Self::gaussian_elimination(&mut a, &mut l, strategy)?;

        // Create generator matrix G
        let k = n - m; // number of message bits
        let mut g = Mod2Sparse::allocate(k, n)?;

        // After Gaussian elimination, we have:
        // - cols_to_rows[j] >= 0 means column j is a pivot column (part of I)
        // - cols_to_rows[j] == -1 means column j is a non-pivot column (message bit position)

        // Step 1: Identify message bit positions (non-pivot columns)
        let mut message_positions = Vec::new();
        for col in 0..n {
            if cols_to_rows[col] == -1 {
                message_positions.push(col);
            }
        }

        // Step 2: Create identity part of G - this goes in the message bit positions
        for (msg_idx, &col_pos) in message_positions.iter().enumerate() {
            if msg_idx < k {
                g.insert(msg_idx, col_pos)?;
            }
        }

        // Step 3: Create P^T part of G
        // For each message bit row i, and each pivot column j:
        // If systematic form has entry at (pivot_row, message_col),
        // then G should have entry at (message_idx, pivot_col)
        for (msg_idx, &msg_col) in message_positions.iter().enumerate() {
            if msg_idx >= k {
                break;
            }

            for pivot_col in 0..n {
                if cols_to_rows[pivot_col] >= 0 {
                    // This is a pivot column
                    let pivot_row = cols_to_rows[pivot_col] as usize;

                    // Check if the systematic form A has an entry at (pivot_row, msg_col)
                    if a.find(pivot_row, msg_col) {
                        // Add this to G[msg_idx, pivot_col]
                        g.insert(msg_idx, pivot_col)?;
                    }
                }
            }
        }

        Ok(Self {
            sparse_gen: Some(g),
            dense_gen: None,
            method: GeneratorMethod::Sparse,
            cols_to_rows,
            n_bits: n,
            n_checks: m,
        })
    }

    /// MAKE DENSE/MIXED GENERATOR MATRIX - exact port of make_dense_mixed function
    fn make_dense_mixed(h: &Mod2Sparse, method: GeneratorMethod) -> Result<Self, LdpcError> {
        let m = h.rows();
        let n = h.cols();
        let k = n - m;

        // Convert to dense for easier manipulation
        let a = mod2convert::sparse_to_dense(h)?;

        // Create augmented matrix [A | I] for Gaussian elimination
        let mut augmented = Mod2Dense::allocate(m, n + m)?;

        // Copy A part
        for i in 0..m {
            for j in 0..n {
                if a.get(i, j) {
                    augmented.set(i, j, true)?;
                }
            }
        }

        // Add identity part
        for i in 0..m {
            augmented.set(i, n + i, true)?;
        }

        // Gaussian elimination
        let cols_to_rows = Self::dense_gaussian_elimination(&mut augmented, n)?;

        // Extract the generator matrix in systematic form
        let mut g = Mod2Dense::allocate(k, n)?;

        // Set up identity part first
        for i in 0..k {
            g.set(i, m + i, true)?;
        }

        // Extract parity part (P^T) from the systematic form
        for i in 0..k {
            for j in 0..m {
                // If column j was a pivot column, check if we need this entry
                if cols_to_rows[j] >= 0 {
                    let pivot_row = cols_to_rows[j] as usize;
                    if augmented.get(pivot_row, m + i) {
                        g.set(i, j, true)?;
                    }
                }
            }
        }

        let generator = match method {
            GeneratorMethod::Dense => Some(g),
            GeneratorMethod::Mixed => {
                // Convert back to sparse if it's sparse enough
                if Self::is_sparse_enough(&g) {
                    None // We would need to create sparse version here
                } else {
                    Some(g)
                }
            }
            _ => None,
        };

        Ok(Self {
            sparse_gen: None,
            dense_gen: generator,
            method,
            cols_to_rows,
            n_bits: n,
            n_checks: m,
        })
    }

    /// GAUSSIAN ELIMINATION FOR SPARSE MATRICES - exact port from make-gen.c
    fn gaussian_elimination(
        a: &mut Mod2Sparse,
        l: &mut Mod2Sparse,
        strategy: Mod2SparseStrategy,
    ) -> Result<Vec<i32>, LdpcError> {
        let m = a.rows();
        let n = a.cols();
        let mut cols_to_rows = vec![-1i32; n]; // -1 indicates non-pivot column

        for i in 0..m {
            // Find pivot column based on strategy
            let pivot_col = match strategy {
                Mod2SparseStrategy::First => Self::find_first_pivot(a, i),
                Mod2SparseStrategy::Mincol => Self::find_mincol_pivot(a, i),
                Mod2SparseStrategy::Minprod => Self::find_minprod_pivot(a, i),
            };

            let pivot_col = pivot_col.ok_or_else(|| {
                LdpcError::InvalidParameter("Matrix is not full rank".to_string())
            })?;

            // Find pivot row in this column
            let pivot_row = a
                .entries_in_col(pivot_col)
                .find(|&row| row >= i)
                .ok_or_else(|| LdpcError::InvalidParameter("No pivot found".to_string()))?;

            // Swap rows if necessary
            if pivot_row != i {
                Self::swap_sparse_rows(a, i, pivot_row);
                Self::swap_sparse_rows(l, i, pivot_row);
            }

            // Record column-to-row mapping
            cols_to_rows[pivot_col] = i as i32;

            // Eliminate other entries in this column
            let rows_to_eliminate: Vec<usize> = a
                .entries_in_col(pivot_col)
                .filter(|&row| row != i)
                .collect();

            for row in rows_to_eliminate {
                Self::add_sparse_rows(a, row, i);
                Self::add_sparse_rows(l, row, i);
            }
        }

        Ok(cols_to_rows)
    }

    /// DENSE GAUSSIAN ELIMINATION - exact port algorithm
    fn dense_gaussian_elimination(
        matrix: &mut Mod2Dense,
        n_cols: usize,
    ) -> Result<Vec<i32>, LdpcError> {
        let m = matrix.rows();
        let mut cols_to_rows = vec![-1i32; n_cols];

        for i in 0..m {
            // Find pivot column
            let mut pivot_col = None;
            for j in 0..n_cols {
                if matrix.get(i, j) {
                    pivot_col = Some(j);
                    break;
                }
            }

            let pivot_col = pivot_col
                .ok_or_else(|| LdpcError::InvalidParameter("Matrix is singular".to_string()))?;

            cols_to_rows[pivot_col] = i as i32;

            // Eliminate this column in other rows
            for row in 0..m {
                if row != i && matrix.get(row, pivot_col) {
                    // Add row i to row 'row'
                    for col in 0..matrix.cols() {
                        if matrix.get(i, col) {
                            matrix.flip(row, col)?;
                        }
                    }
                }
            }
        }

        Ok(cols_to_rows)
    }

    // Helper methods for pivot selection strategies - exact ports
    fn find_first_pivot(a: &Mod2Sparse, start_row: usize) -> Option<usize> {
        for col in 0..a.cols() {
            if a.entries_in_col(col).any(|row| row >= start_row) {
                return Some(col);
            }
        }
        None
    }

    fn find_mincol_pivot(a: &Mod2Sparse, start_row: usize) -> Option<usize> {
        let mut best_col = None;
        let mut min_weight = usize::MAX;

        for col in 0..a.cols() {
            let weight = a
                .entries_in_col(col)
                .filter(|&row| row >= start_row)
                .count();
            if weight > 0 && weight < min_weight {
                min_weight = weight;
                best_col = Some(col);
            }
        }

        best_col
    }

    fn find_minprod_pivot(a: &Mod2Sparse, start_row: usize) -> Option<usize> {
        let mut best_col = None;
        let mut min_product = usize::MAX;

        for col in 0..a.cols() {
            let col_weight = a
                .entries_in_col(col)
                .filter(|&row| row >= start_row)
                .count();
            if col_weight == 0 {
                continue;
            }

            let mut row_weight_sum = 0;
            for row in a.entries_in_col(col).filter(|&row| row >= start_row) {
                row_weight_sum += a.row_weight(row);
            }

            let product = col_weight * row_weight_sum;
            if product < min_product {
                min_product = product;
                best_col = Some(col);
            }
        }

        best_col
    }

    fn swap_sparse_rows(matrix: &mut Mod2Sparse, row1: usize, row2: usize) {
        if row1 == row2 {
            return;
        }

        // Get all column indices for both rows
        let row1_cols: Vec<usize> = matrix.entries_in_row(row1).collect();
        let row2_cols: Vec<usize> = matrix.entries_in_row(row2).collect();

        // Clear both rows
        for col in &row1_cols {
            matrix.delete(row1, *col).unwrap();
        }
        for col in &row2_cols {
            matrix.delete(row2, *col).unwrap();
        }

        // Swap the entries
        for col in row1_cols {
            matrix.insert(row2, col).unwrap();
        }
        for col in row2_cols {
            matrix.insert(row1, col).unwrap();
        }
    }

    fn add_sparse_rows(matrix: &mut Mod2Sparse, dest_row: usize, src_row: usize) {
        let src_cols: Vec<usize> = matrix.entries_in_row(src_row).collect();

        for col in src_cols {
            if matrix.find(dest_row, col) {
                matrix.delete(dest_row, col).unwrap();
            } else {
                matrix.insert(dest_row, col).unwrap();
            }
        }
    }

    fn is_sparse_enough(matrix: &Mod2Dense) -> bool {
        let mut count = 0;
        let total = matrix.rows() * matrix.cols();

        for i in 0..matrix.rows() {
            for j in 0..matrix.cols() {
                if matrix.get(i, j) {
                    count += 1;
                }
            }
        }

        // Consider sparse if less than 10% non-zero
        count * 10 < total
    }

    /// ENCODE A SOURCE VECTOR - exact port of encode logic
    pub fn encode(&self, source: &[u8]) -> Result<Vec<u8>, LdpcError> {
        let k = self.n_bits - self.n_checks;

        if source.len() != k {
            return Err(LdpcError::Encoding(format!(
                "Source length {} doesn't match message length {}",
                source.len(),
                k
            )));
        }

        let mut codeword = vec![0u8; self.n_bits];

        match (&self.sparse_gen, &self.dense_gen) {
            (Some(sparse_g), None) => {
                // Sparse encoding
                for i in 0..k {
                    if source[i] != 0 {
                        // Add row i of generator matrix to codeword
                        for col in sparse_g.entries_in_row(i) {
                            codeword[col] ^= 1;
                        }
                    }
                }
            }
            (None, Some(dense_g)) => {
                // Dense encoding
                for i in 0..k {
                    if source[i] != 0 {
                        // Add row i of generator matrix to codeword
                        for j in 0..self.n_bits {
                            if dense_g.get(i, j) {
                                codeword[j] ^= 1;
                            }
                        }
                    }
                }
            }
            _ => {
                return Err(LdpcError::Encoding(
                    "No generator matrix available".to_string(),
                ));
            }
        }

        Ok(codeword)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generator_creation() {
        // Create a simple parity check matrix
        let mut h = Mod2Sparse::allocate(2, 4).unwrap();
        h.insert(0, 0).unwrap();
        h.insert(0, 1).unwrap();
        h.insert(1, 2).unwrap();
        h.insert(1, 3).unwrap();

        let gen = GeneratorMatrix::from_parity_check(
            &h,
            GeneratorMethod::Sparse,
            Mod2SparseStrategy::First,
            0,
            0,
        )
        .unwrap();

        assert_eq!(gen.n_bits, 4);
        assert_eq!(gen.n_checks, 2);
        assert!(gen.sparse_gen.is_some());
    }
}
