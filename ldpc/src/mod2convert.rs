use crate::{LdpcError, Mod2Dense, Mod2Sparse};

/// CONVERT A MOD2 MATRIX FROM SPARSE TO DENSE FORM - exact port of mod2sparse_to_dense
pub fn sparse_to_dense(sparse: &Mod2Sparse) -> Result<Mod2Dense, LdpcError> {
    let mut dense = Mod2Dense::allocate(sparse.rows(), sparse.cols())?;

    for row in 0..sparse.rows() {
        for col in sparse.entries_in_row(row) {
            dense.set(row, col, true)?;
        }
    }

    Ok(dense)
}

/// CONVERT A MOD2 MATRIX FROM DENSE TO SPARSE FORM - exact port of mod2dense_to_sparse
pub fn dense_to_sparse(dense: &Mod2Dense) -> Result<Mod2Sparse, LdpcError> {
    let mut sparse = Mod2Sparse::allocate(dense.rows(), dense.cols())?;

    for row in 0..dense.rows() {
        for col in 0..dense.cols() {
            if dense.get(row, col) {
                sparse.insert(row, col)?;
            }
        }
    }

    Ok(sparse)
}

/// FIND THE NUMBER OF 1s IN A MOD2 SPARSE MATRIX - exact port of mod2sparse_count
pub fn sparse_count_ones(matrix: &Mod2Sparse) -> usize {
    let mut count = 0;
    for row in 0..matrix.rows() {
        count += matrix.row_weight(row);
    }
    count
}

/// FIND THE NUMBER OF 1s IN A MOD2 DENSE MATRIX - exact port of mod2dense_count
pub fn dense_count_ones(matrix: &Mod2Dense) -> usize {
    let mut count = 0;
    for row in 0..matrix.rows() {
        for col in 0..matrix.cols() {
            if matrix.get(row, col) {
                count += 1;
            }
        }
    }
    count
}

/// Check if two matrices are equal
pub fn matrices_equal(sparse: &Mod2Sparse, dense: &Mod2Dense) -> bool {
    if sparse.rows() != dense.rows() || sparse.cols() != dense.cols() {
        return false;
    }

    for row in 0..sparse.rows() {
        for col in 0..sparse.cols() {
            if sparse.find(row, col) != dense.get(row, col) {
                return false;
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conversion_roundtrip() {
        // Create a sparse matrix
        let mut sparse = Mod2Sparse::allocate(3, 4).unwrap();
        sparse.insert(0, 1).unwrap();
        sparse.insert(1, 2).unwrap();
        sparse.insert(2, 0).unwrap();
        sparse.insert(2, 3).unwrap();

        // Convert to dense
        let dense = sparse_to_dense(&sparse).unwrap();

        // Check conversion
        assert!(dense.get(0, 1));
        assert!(dense.get(1, 2));
        assert!(dense.get(2, 0));
        assert!(dense.get(2, 3));
        assert!(!dense.get(0, 0));

        // Convert back to sparse
        let sparse2 = dense_to_sparse(&dense).unwrap();

        // Check they're equivalent
        assert!(matrices_equal(&sparse, &dense));
        assert!(matrices_equal(&sparse2, &dense));
    }

    #[test]
    fn test_count_ones() {
        let mut sparse = Mod2Sparse::allocate(3, 3).unwrap();
        sparse.insert(0, 0).unwrap();
        sparse.insert(1, 1).unwrap();
        sparse.insert(2, 2).unwrap();

        let dense = sparse_to_dense(&sparse).unwrap();

        assert_eq!(sparse_count_ones(&sparse), 3);
        assert_eq!(dense_count_ones(&dense), 3);
    }
}
