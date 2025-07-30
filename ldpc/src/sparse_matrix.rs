use std::collections::HashMap;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum SparseError {
    #[error("Invalid matrix dimensions")]
    InvalidDimensions,
    #[error("Row or column index out of bounds")]
    IndexOutOfBounds,
    #[error("Entry not found")]
    EntryNotFound,
}

/// Entry in sparse matrix - Rust-friendly version of mod2entry
#[derive(Debug, Clone)]
pub struct Mod2Entry {
    pub row: usize,
    pub col: usize,
    pub pr: f64, // Probability ratio for belief propagation
    pub lr: f64, // Likelihood ratio for belief propagation
}

/// Sparse matrix over GF(2) - Rust-friendly while maintaining exact algorithms
#[derive(Debug, Clone)]
pub struct Mod2Sparse {
    n_rows: usize,
    n_cols: usize,
    // Use HashMap for fast lookup, Vec<Vec<usize>> for ordered iteration
    // This maintains the same algorithmic behavior as the linked lists
    row_entries: Vec<Vec<usize>>, // For each row, list of column indices
    col_entries: Vec<Vec<usize>>, // For each col, list of row indices
    entries: HashMap<(usize, usize), Mod2Entry>, // Actual entry storage
}

impl Mod2Sparse {
    /// ALLOCATE SPACE FOR A SPARSE MOD2 MATRIX - exact algorithm, Rust-friendly implementation
    pub fn allocate(n_rows: usize, n_cols: usize) -> Result<Self, SparseError> {
        if n_rows == 0 || n_cols == 0 {
            return Err(SparseError::InvalidDimensions);
        }

        Ok(Self {
            n_rows,
            n_cols,
            row_entries: vec![Vec::new(); n_rows],
            col_entries: vec![Vec::new(); n_cols],
            entries: HashMap::new(),
        })
    }

    pub fn rows(&self) -> usize {
        self.n_rows
    }
    pub fn cols(&self) -> usize {
        self.n_cols
    }

    /// LOOK FOR AN ENTRY WITH GIVEN ROW AND COLUMN - exact port of mod2sparse_find
    pub fn find(&self, row: usize, col: usize) -> bool {
        self.entries.contains_key(&(row, col))
    }

    /// INSERT AN ENTRY - exact port of mod2sparse_insert algorithm
    pub fn insert(&mut self, row: usize, col: usize) -> Result<(), SparseError> {
        if row >= self.n_rows || col >= self.n_cols {
            return Err(SparseError::IndexOutOfBounds);
        }

        if self.find(row, col) {
            return Ok(()); // Already exists
        }

        // Insert maintaining sorted order (same as original linked list behavior)
        let row_pos = self.row_entries[row]
            .binary_search(&col)
            .unwrap_or_else(|x| x);
        self.row_entries[row].insert(row_pos, col);

        let col_pos = self.col_entries[col]
            .binary_search(&row)
            .unwrap_or_else(|x| x);
        self.col_entries[col].insert(col_pos, row);

        self.entries.insert(
            (row, col),
            Mod2Entry {
                row,
                col,
                pr: 0.0,
                lr: 0.0,
            },
        );

        Ok(())
    }

    /// DELETE AN ENTRY - exact port of mod2sparse_delete algorithm
    pub fn delete(&mut self, row: usize, col: usize) -> Result<(), SparseError> {
        if !self.find(row, col) {
            return Err(SparseError::EntryNotFound);
        }

        // Remove from row list
        if let Some(pos) = self.row_entries[row].iter().position(|&x| x == col) {
            self.row_entries[row].remove(pos);
        }

        // Remove from column list
        if let Some(pos) = self.col_entries[col].iter().position(|&x| x == row) {
            self.col_entries[col].remove(pos);
        }

        // Remove from entries
        self.entries.remove(&(row, col));

        Ok(())
    }

    /// Get entry reference for belief propagation values
    pub fn get_entry_mut(&mut self, row: usize, col: usize) -> Option<&mut Mod2Entry> {
        self.entries.get_mut(&(row, col))
    }

    pub fn get_entry(&self, row: usize, col: usize) -> Option<&Mod2Entry> {
        self.entries.get(&(row, col))
    }

    /// Iterate over entries in a row - maintains same order as original
    pub fn entries_in_row(&self, row: usize) -> impl Iterator<Item = usize> + '_ {
        self.row_entries[row].iter().copied()
    }

    /// Iterate over entries in a column - maintains same order as original
    pub fn entries_in_col(&self, col: usize) -> impl Iterator<Item = usize> + '_ {
        self.col_entries[col].iter().copied()
    }

    /// Check if at first entry in row (for algorithm ports)
    pub fn first_in_row(&self, row: usize) -> Option<usize> {
        self.row_entries[row].first().copied()
    }

    /// Check if at first entry in column (for algorithm ports)
    pub fn first_in_col(&self, col: usize) -> Option<usize> {
        self.col_entries[col].first().copied()
    }

    /// Get next entry in row after given column
    pub fn next_in_row(&self, row: usize, col: usize) -> Option<usize> {
        let pos = self.row_entries[row].iter().position(|&x| x == col)?;
        self.row_entries[row].get(pos + 1).copied()
    }

    /// Get next entry in column after given row
    pub fn next_in_col(&self, col: usize, row: usize) -> Option<usize> {
        let pos = self.col_entries[col].iter().position(|&x| x == row)?;
        self.col_entries[col].get(pos + 1).copied()
    }

    /// CLEAR A SPARSE MATRIX TO ALL ZEROS - exact port of mod2sparse_clear
    pub fn clear(&mut self) {
        for row_list in &mut self.row_entries {
            row_list.clear();
        }
        for col_list in &mut self.col_entries {
            col_list.clear();
        }
        self.entries.clear();
    }

    /// Get weight (number of 1s) in a row
    pub fn row_weight(&self, row: usize) -> usize {
        self.row_entries[row].len()
    }

    /// Get weight (number of 1s) in a column
    pub fn col_weight(&self, col: usize) -> usize {
        self.col_entries[col].len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_operations() {
        let mut matrix = Mod2Sparse::allocate(3, 4).unwrap();

        // Test insertion
        matrix.insert(0, 1).unwrap();
        matrix.insert(1, 2).unwrap();
        matrix.insert(0, 3).unwrap();

        // Test finding
        assert!(matrix.find(0, 1));
        assert!(matrix.find(1, 2));
        assert!(!matrix.find(2, 0));

        // Test iteration maintains order
        let row0_cols: Vec<usize> = matrix.entries_in_row(0).collect();
        assert_eq!(row0_cols, vec![1, 3]); // Should be sorted

        // Test deletion
        matrix.delete(0, 1).unwrap();
        assert!(!matrix.find(0, 1));
        assert!(matrix.find(0, 3));
    }
}
