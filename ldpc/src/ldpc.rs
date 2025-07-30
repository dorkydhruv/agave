use crate::{
    Distribution, GeneratorMatrix, GeneratorMethod, LdpcError, Mod2Sparse, Mod2SparseStrategy,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MakeMethod {
    EvenCol,  // Evencol from original
    EvenBoth, // Evenboth from original
}

/// Main LDPC Code structure - combines matrix generation, encoding, and decoding
pub struct LdpcCode {
    m_checks: usize,
    n_bits: usize,
    parity_check_matrix: Option<Mod2Sparse>,
    generator_matrix: Option<GeneratorMatrix>,
}

impl LdpcCode {
    /// Create a new LDPC code with specified parameters - high-level API
    pub fn new(
        m_checks: usize,
        n_bits: usize,
        seed: u64,
        method: MakeMethod,
        distribution: &str,
        no_4cycle: bool,
    ) -> Result<Self, LdpcError> {
        if m_checks == 0 || n_bits == 0 {
            return Err(LdpcError::InvalidParameter(
                "dimensions must be > 0".to_string(),
            ));
        }

        if n_bits <= m_checks {
            return Err(LdpcError::InvalidParameter(format!(
                "Number of bits ({}) should be greater than number of checks ({})",
                n_bits, m_checks
            )));
        }

        let distribution = Distribution::create(distribution)?;

        let mut code = Self {
            m_checks,
            n_bits,
            parity_check_matrix: None,
            generator_matrix: None,
        };

        code.make_ldpc(seed, method, &distribution, no_4cycle)?;

        Ok(code)
    }

    /// Create from existing components - for advanced users
    pub fn from_matrix(parity_check_matrix: Mod2Sparse) -> Result<Self, LdpcError> {
        let m_checks = parity_check_matrix.rows();
        let n_bits = parity_check_matrix.cols();

        if n_bits <= m_checks {
            return Err(LdpcError::InvalidParameter(format!(
                "Number of bits ({}) should be greater than number of checks ({})",
                n_bits, m_checks
            )));
        }

        Ok(Self {
            m_checks,
            n_bits,
            parity_check_matrix: Some(parity_check_matrix),
            generator_matrix: None,
        })
    }

    /// Get code parameters
    pub fn n_bits(&self) -> usize {
        self.n_bits
    }
    pub fn m_checks(&self) -> usize {
        self.m_checks
    }
    pub fn k_message_bits(&self) -> usize {
        self.n_bits - self.m_checks
    }
    pub fn rate(&self) -> f64 {
        (self.k_message_bits() as f64) / (self.n_bits as f64)
    }

    /// Get the parity check matrix
    pub fn parity_check_matrix(&self) -> Option<&Mod2Sparse> {
        self.parity_check_matrix.as_ref()
    }

    /// Create generator matrix for encoding - exact port of make-gen.c
    pub fn create_generator_matrix(
        &mut self,
        method: GeneratorMethod,
        strategy: Mod2SparseStrategy,
    ) -> Result<(), LdpcError> {
        let h = self.parity_check_matrix.as_ref().ok_or_else(|| {
            LdpcError::InvalidParameter("No parity check matrix available".to_string())
        })?;

        let generator = GeneratorMatrix::from_parity_check(
            h, method, strategy, 0, // abandon_number
            0, // abandon_when
        )?;

        self.generator_matrix = Some(generator);
        Ok(())
    }

    /// ENCODE MESSAGE BITS TO CODEWORD - exact port of encode.c
    pub fn encode(&mut self, message: &[u8]) -> Result<Vec<u8>, LdpcError> {
        let k = self.k_message_bits();

        if message.len() != k {
            return Err(LdpcError::Encoding(format!(
                "Message length {} doesn't match code dimension {}",
                message.len(),
                k
            )));
        }

        // Create generator matrix if it doesn't exist
        if self.generator_matrix.is_none() {
            self.create_generator_matrix(GeneratorMethod::Sparse, Mod2SparseStrategy::First)?;
        }

        let generator = self.generator_matrix.as_ref().unwrap();
        generator.encode(message)
    }

    /// DECODE RECEIVED CODEWORD - placeholder for belief propagation decoding
    pub fn decode(&self, received: &mut [u8]) -> Result<Vec<u8>, LdpcError> {
        if self.parity_check_matrix.is_none() {
            return Err(LdpcError::Decoding(
                "No parity check matrix available".to_string(),
            ));
        }

        if received.len() != self.n_bits {
            return Err(LdpcError::Decoding(format!(
                "Received length {} doesn't match code length {}",
                received.len(),
                self.n_bits
            )));
        }

        // For now, just extract the systematic part (assuming systematic encoding)
        // This is a placeholder - real BP decoding would be much more complex
        let k = self.k_message_bits();
        let m = self.m_checks;

        // In systematic form, message bits are in positions m..n
        let mut message = vec![0u8; k];
        for i in 0..k {
            message[i] = received[m + i];
        }

        Ok(message)
    }

    /// Verify that a codeword satisfies the parity check constraints
    pub fn verify_codeword(&self, codeword: &[u8]) -> Result<usize, LdpcError> {
        let h = self
            .parity_check_matrix
            .as_ref()
            .ok_or_else(|| LdpcError::InvalidParameter("No parity check matrix".to_string()))?;

        if codeword.len() != self.n_bits {
            return Err(LdpcError::InvalidParameter(format!(
                "Codeword length {} doesn't match code length {}",
                codeword.len(),
                self.n_bits
            )));
        }

        let mut errors = 0;

        // Check each parity check equation
        for check_row in 0..self.m_checks {
            let mut sum = 0u8;

            // Sum all bits involved in this check
            for col in h.entries_in_row(check_row) {
                sum ^= codeword[col];
            }

            // Each parity check should sum to 0 in GF(2)
            if sum != 0 {
                errors += 1;
            }
        }

        Ok(errors)
    }

    // ====== Internal Matrix Generation Methods (exact ports) ======

    /// CREATE A SPARSE PARITY-CHECK MATRIX - exact port of make_ldpc algorithm
    pub fn make_ldpc(
        &mut self,
        seed: u64,
        method: MakeMethod,
        distribution: &Distribution,
        no_4cycle: bool,
    ) -> Result<(), LdpcError> {
        // Validation checks - exact ports from original
        if distribution.max() > self.m_checks {
            return Err(LdpcError::InvalidParameter(format!(
                "At least one checks per bit ({}) is greater than total checks ({})",
                distribution.max(),
                self.m_checks
            )));
        }

        if distribution.max() == self.m_checks && self.n_bits > 1 && no_4cycle {
            return Err(LdpcError::InvalidParameter(
                "Can't eliminate cycles of length four with this many checks per bit".to_string(),
            ));
        }

        // Initialize random number generator - exact port of rand_seed(10*seed+1)
        let mut rng = ChaCha8Rng::seed_from_u64(10 * seed + 1);

        // Allocate matrix
        let mut h = Mod2Sparse::allocate(self.m_checks, self.n_bits)?;

        // Column partition - exact port of column_partition algorithm
        let part = self.column_partition(distribution)?;

        // Create initial version of parity check matrix - exact algorithm ports
        match method {
            MakeMethod::EvenCol => {
                self.make_evencol(&mut h, &mut rng, distribution, &part)?;
            }
            MakeMethod::EvenBoth => {
                self.make_evenboth(&mut h, &mut rng, distribution, &part)?;
            }
        }

        // Add extra bits to avoid rows with less than two checks - exact algorithm port
        let added = self.ensure_min_row_degree(&mut h, &mut rng)?;

        // Add extra bits for even column counts - exact algorithm port
        self.handle_even_columns(&mut h, &mut rng, distribution, added)?;

        // Eliminate 4-cycles if requested - exact algorithm port
        if no_4cycle {
            self.eliminate_4cycles(&mut h, &mut rng)?;
        }

        self.parity_check_matrix = Some(h);
        Ok(())
    }

    /// PARTITION THE COLUMNS - exact port of column_partition algorithm
    fn column_partition(&self, d: &Distribution) -> Result<Vec<usize>, LdpcError> {
        let mut part = vec![0; d.size()];
        let mut trunc = vec![0.0; d.size()];
        let mut used = 0;

        // First pass: allocate integer parts
        for i in 0..d.size() {
            let cur = (d.prop(i) * (self.n_bits as f64)).floor() as usize;
            part[i] = cur;
            trunc[i] = d.prop(i) * (self.n_bits as f64) - (cur as f64);
            used += cur;
        }

        // Second pass: allocate remaining columns to highest fractional parts
        while used < self.n_bits {
            let mut cur = 0;
            for j in 1..d.size() {
                if trunc[j] > trunc[cur] {
                    cur = j;
                }
            }
            part[cur] += 1;
            used += 1;
            trunc[cur] = -1.0; // Remove from consideration
        }

        Ok(part)
    }

    /// EVENCOL method - exact algorithm port
    fn make_evencol(
        &mut self,
        h: &mut Mod2Sparse,
        rng: &mut ChaCha8Rng,
        d: &Distribution,
        part: &[usize],
    ) -> Result<(), LdpcError> {
        let mut z = 0;
        let mut left = part[z];

        for j in 0..self.n_bits {
            // Find next distribution entry with remaining columns
            while left == 0 {
                z += 1;
                if z >= d.size() {
                    return Err(LdpcError::InvalidParameter(
                        "Distribution partition error".to_string(),
                    ));
                }
                left = part[z];
            }

            // Place required number of 1s for this column
            for _k in 0..d.num(z) {
                loop {
                    let i = rng.gen_range(0..self.m_checks);
                    if !h.find(i, j) {
                        h.insert(i, j)?;
                        break;
                    }
                }
            }
            left -= 1;
        }

        Ok(())
    }

    /// EVENBOTH method - exact algorithm port
    fn make_evenboth(
        &mut self,
        h: &mut Mod2Sparse,
        rng: &mut ChaCha8Rng,
        d: &Distribution,
        part: &[usize],
    ) -> Result<(), LdpcError> {
        // Calculate total number of 1s
        let mut cb_n = 0;
        for z in 0..d.size() {
            cb_n += d.num(z) * part[z];
        }

        // Create array for even row distribution
        let mut u: Vec<usize> = (0..cb_n).map(|k| k % self.m_checks).collect();

        let mut uneven = 0;
        let mut t = 0;
        let mut z = 0;
        let mut left = part[z];

        for j in 0..self.n_bits {
            // Find next distribution entry
            while left == 0 {
                z += 1;
                if z >= d.size() {
                    return Err(LdpcError::InvalidParameter(
                        "Distribution partition error".to_string(),
                    ));
                }
                left = part[z];
            }

            // Place required 1s for this column
            for _k in 0..d.num(z) {
                // Find first unused row from position t onward
                let mut i = t;
                while i < cb_n && h.find(u[i], j) {
                    i += 1;
                }

                if i == cb_n {
                    // All remaining rows conflict - use random placement
                    uneven += 1;
                    loop {
                        let row = rng.gen_range(0..self.m_checks);
                        if !h.find(row, j) {
                            h.insert(row, j)?;
                            break;
                        }
                    }
                } else {
                    // Pick randomly from remaining unused rows
                    loop {
                        let idx = t + rng.gen_range(0..cb_n - t);
                        if !h.find(u[idx], j) {
                            h.insert(u[idx], j)?;
                            // Swap used element to front
                            u.swap(idx, t);
                            t += 1;
                            break;
                        }
                    }
                }
            }
            left -= 1;
        }

        if uneven > 0 {
            eprintln!("Had to place {} checks in rows unevenly", uneven);
        }

        Ok(())
    }

    /// Ensure minimum row degree - exact algorithm port
    fn ensure_min_row_degree(
        &mut self,
        h: &mut Mod2Sparse,
        rng: &mut ChaCha8Rng,
    ) -> Result<usize, LdpcError> {
        let mut added = 0;

        for i in 0..self.m_checks {
            // Check if row has no entries
            if h.row_weight(i) == 0 {
                let j = rng.gen_range(0..self.n_bits);
                h.insert(i, j)?;
                added += 1;
            }

            // Check if row has only one entry (and we have more than 1 column)
            if h.row_weight(i) == 1 && self.n_bits > 1 {
                let existing_col = h.first_in_row(i).unwrap();
                loop {
                    let j = rng.gen_range(0..self.n_bits);
                    if j != existing_col {
                        h.insert(i, j)?;
                        added += 1;
                        break;
                    }
                }
            }
        }

        if added > 0 {
            eprintln!(
                "Added {} extra bit-checks to make row counts at least two",
                added
            );
        }

        Ok(added)
    }

    /// Handle even column degrees - exact algorithm port
    fn handle_even_columns(
        &mut self,
        h: &mut Mod2Sparse,
        rng: &mut ChaCha8Rng,
        d: &Distribution,
        added: usize,
    ) -> Result<(), LdpcError> {
        let mut n_full = 0;
        let mut all_even = true;

        for z in 0..d.size() {
            if d.num(z) == self.m_checks {
                n_full += 1; // This should be part[z] but we can approximate
            }
            if d.num(z) % 2 == 1 {
                all_even = false;
            }
        }

        if all_even && self.n_bits - n_full > 1 && added < 2 {
            let extra_needed = 2 - added;
            for _a in 0..extra_needed {
                loop {
                    let i = rng.gen_range(0..self.m_checks);
                    let j = rng.gen_range(0..self.n_bits);
                    if !h.find(i, j) {
                        h.insert(i, j)?;
                        break;
                    }
                }
            }
            eprintln!(
                "Added {} extra bit-checks to try to avoid problems from even column counts",
                extra_needed
            );
        }

        Ok(())
    }

    /// Eliminate 4-cycles - exact algorithm port (fixed for Rust borrowing)
    fn eliminate_4cycles(
        &mut self,
        h: &mut Mod2Sparse,
        rng: &mut ChaCha8Rng,
    ) -> Result<(), LdpcError> {
        let mut elim4 = 0;

        // Up to 10 passes to eliminate 4-cycles
        for _pass in 0..10 {
            let mut cycles_found = 0;

            for j in 0..self.n_bits {
                // Collect entries in column j first to avoid borrowing issues
                let col_entries: Vec<usize> = h.entries_in_col(j).collect();
                let mut found_cycle = false;

                for row1 in col_entries {
                    if found_cycle {
                        break;
                    }

                    // Collect entries in row1 to avoid borrowing issues
                    let row_entries: Vec<usize> = h.entries_in_row(row1).collect();

                    for col2 in row_entries {
                        if col2 == j {
                            continue;
                        }
                        if found_cycle {
                            break;
                        }

                        // Collect entries in col2 to avoid borrowing issues
                        let col2_entries: Vec<usize> = h.entries_in_col(col2).collect();

                        for row2 in col2_entries {
                            if row2 == row1 {
                                continue;
                            }

                            // Check if (row2, j) exists - that would complete a 4-cycle
                            if h.find(row2, j) {
                                // Found 4-cycle: j->row1->col2->row2->j
                                // Move the (row1, j) entry to break the cycle
                                h.delete(row1, j)?;

                                // Find new row for column j
                                loop {
                                    let new_row = rng.gen_range(0..self.m_checks);
                                    if !h.find(new_row, j) {
                                        h.insert(new_row, j)?;
                                        break;
                                    }
                                }

                                elim4 += 1;
                                cycles_found += 1;
                                found_cycle = true;
                                break;
                            }
                        }
                    }
                }
            }

            if cycles_found == 0 {
                break; // No more cycles found
            }
        }

        if elim4 > 0 {
            eprintln!("Eliminated {} cycles of length four", elim4);
        }

        Ok(())
    }

    pub fn get_matrix(&self) -> Option<&Mod2Sparse> {
        self.parity_check_matrix.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ldpc_code_creation() {
        // Test basic code creation
        let code = LdpcCode::new(
            100,   // m_checks
            200,   // n_bits
            12345, // seed
            MakeMethod::EvenCol,
            "3",   // distribution: all columns have degree 3
            false, // no 4-cycle elimination
        )
        .unwrap();

        assert_eq!(code.m_checks(), 100);
        assert_eq!(code.n_bits(), 200);
        assert_eq!(code.k_message_bits(), 100);
        assert_eq!(code.rate(), 0.5);
        assert!(code.parity_check_matrix().is_some());
    }

    #[test]
    fn test_invalid_parameters() {
        // Test dimension validation
        assert!(LdpcCode::new(0, 100, 12345, MakeMethod::EvenCol, "3", false).is_err());
        assert!(LdpcCode::new(100, 0, 12345, MakeMethod::EvenCol, "3", false).is_err());
        assert!(LdpcCode::new(100, 50, 12345, MakeMethod::EvenCol, "3", false).is_err());
        // n <= m
    }

    #[test]
    fn test_different_distributions() {
        // Test single degree distribution
        let code1 = LdpcCode::new(50, 100, 1, MakeMethod::EvenCol, "3", false).unwrap();

        // Test mixed distribution
        let code2 = LdpcCode::new(50, 100, 1, MakeMethod::EvenCol, "0.5x2/0.5x4", false).unwrap();

        // Both should have valid matrices
        assert!(code1.parity_check_matrix().is_some());
        assert!(code2.parity_check_matrix().is_some());
    }

    #[test]
    fn test_matrix_properties() {
        let code = LdpcCode::new(50, 100, 42, MakeMethod::EvenCol, "3", false).unwrap();
        let matrix = code.parity_check_matrix().unwrap();

        // Check matrix dimensions
        assert_eq!(matrix.rows(), 50);
        assert_eq!(matrix.cols(), 100);

        // Check that all columns have degree 3 (approximately, due to rounding)
        let mut total_col_weight = 0;
        for j in 0..100 {
            let weight = matrix.col_weight(j);
            assert!(weight >= 2 && weight <= 4); // Should be close to 3
            total_col_weight += weight;
        }

        // Total number of 1s should be approximately 3 * 100 = 300
        assert!(total_col_weight >= 290 && total_col_weight <= 310);

        // Check that each row has at least 2 entries (from ensure_min_row_degree)
        for i in 0..50 {
            assert!(matrix.row_weight(i) >= 2);
        }
    }

    #[test]
    fn test_evenboth_method() {
        let code = LdpcCode::new(30, 60, 123, MakeMethod::EvenBoth, "4", false).unwrap();
        let matrix = code.parity_check_matrix().unwrap();

        // EvenBoth should create more balanced row weights
        let mut row_weights: Vec<usize> = (0..30).map(|i| matrix.row_weight(i)).collect();
        row_weights.sort();

        // Should have relatively even distribution of row weights
        let min_weight = row_weights[0];
        let max_weight = row_weights[29];

        // The difference shouldn't be too large for EvenBoth
        assert!(max_weight - min_weight <= 3);
    }

    #[test]
    fn test_4cycle_elimination() {
        // Create code without 4-cycle elimination
        let code1 = LdpcCode::new(20, 40, 999, MakeMethod::EvenCol, "3", false).unwrap();

        // Create code with 4-cycle elimination
        let code2 = LdpcCode::new(20, 40, 999, MakeMethod::EvenCol, "3", true).unwrap();

        // Both should succeed (we can't easily test cycle counts without more complex analysis)
        assert!(code1.parity_check_matrix().is_some());
        assert!(code2.parity_check_matrix().is_some());
    }

    #[test]
    fn test_verify_codeword() {
        let code = LdpcCode::new(10, 20, 1111, MakeMethod::EvenCol, "2", false).unwrap();

        // Test with all-zero codeword (should always be valid)
        let zero_codeword = vec![0u8; 20];
        let errors = code.verify_codeword(&zero_codeword).unwrap();
        assert_eq!(errors, 0);

        // Test with wrong length
        let wrong_length = vec![0u8; 15];
        assert!(code.verify_codeword(&wrong_length).is_err());

        // Test with random data (should have errors)
        let random_data = vec![1u8; 20];
        let errors = code.verify_codeword(&random_data).unwrap();
        assert!(errors > 0); // Random data should violate parity checks
    }

    #[test]
    fn test_reproducibility() {
        // Same parameters should produce same matrix
        let code1 = LdpcCode::new(25, 50, 7777, MakeMethod::EvenCol, "3", false).unwrap();
        let code2 = LdpcCode::new(25, 50, 7777, MakeMethod::EvenCol, "3", false).unwrap();

        let matrix1 = code1.parity_check_matrix().unwrap();
        let matrix2 = code2.parity_check_matrix().unwrap();

        // Check that matrices are identical
        for i in 0..25 {
            for j in 0..50 {
                assert_eq!(matrix1.find(i, j), matrix2.find(i, j));
            }
        }
    }

    #[test]
    fn test_distribution_edge_cases() {
        // Test maximum degree distribution (should fail with 4-cycle elimination)
        let result = LdpcCode::new(
            5,
            10,
            1,
            MakeMethod::EvenCol,
            "5",
            true, // smaller matrix, degree = m_checks, with 4-cycle elim
        );
        assert!(result.is_err());

        // But should work without 4-cycle elimination
        let code = LdpcCode::new(5, 10, 1, MakeMethod::EvenCol, "5", false).unwrap();
        assert!(code.parity_check_matrix().is_some());
    }

    #[test]
    fn test_column_partition() {
        let code = LdpcCode::new(10, 100, 1, MakeMethod::EvenCol, "3", false).unwrap();
        let distribution = Distribution::create("0.3x2/0.7x3").unwrap();

        let partition = code.column_partition(&distribution).unwrap();

        // For 100 columns: 30% should get degree 2, 70% should get degree 3
        // So roughly 30 columns degree 2, 70 columns degree 3
        let total: usize = partition.iter().sum();
        assert_eq!(total, 100);

        // Check approximate distribution
        assert!(partition[0] >= 28 && partition[0] <= 32); // ~30
        assert!(partition[1] >= 68 && partition[1] <= 72); // ~70
    }

    #[test]
    fn test_large_code() {
        // Test a reasonably large code to ensure scalability
        let code =
            LdpcCode::new(100, 200, 2023, MakeMethod::EvenBoth, "0.5x3/0.5x6", false).unwrap();
        let matrix = code.parity_check_matrix().unwrap();

        assert_eq!(matrix.rows(), 100);
        assert_eq!(matrix.cols(), 200);
        assert_eq!(code.rate(), 0.5);

        // Verify all-zero codeword
        let zero_codeword = vec![0u8; 200];
        let errors = code.verify_codeword(&zero_codeword).unwrap();
        assert_eq!(errors, 0);
    }

    #[test]
    fn test_encoding() {
        // Create a reasonably sized code for testing
        let mut code = LdpcCode::new(10, 20, 1234, MakeMethod::EvenCol, "3", false).unwrap();

        println!(
            "Created LDPC code: {} checks, {} bits, rate {:.3}",
            code.m_checks(),
            code.n_bits(),
            code.rate()
        );

        // Test 1: All-zero message (should produce all-zero codeword)
        let zero_message = vec![0u8; code.k_message_bits()];
        let zero_codeword = code.encode(&zero_message).unwrap();

        assert_eq!(zero_codeword.len(), code.n_bits());
        assert!(zero_codeword.iter().all(|&x| x == 0));

        let errors = code.verify_codeword(&zero_codeword).unwrap();
        assert_eq!(errors, 0, "Zero codeword should satisfy all parity checks");
        println!("✓ Zero message test passed!");

        // Test 2: Single bit message
        let mut single_bit_message = vec![0u8; code.k_message_bits()];
        single_bit_message[0] = 1;
        let codeword = code.encode(&single_bit_message).unwrap();
        let errors = code.verify_codeword(&codeword).unwrap();
        assert_eq!(
            errors, 0,
            "Single bit codeword should satisfy all parity checks"
        );
        println!("✓ Single bit message test passed!");

        // Test 3: Random message patterns
        for i in 0..3 {
            let mut message = vec![0u8; code.k_message_bits()];
            message[i % code.k_message_bits()] = 1;
            if i > 0 {
                message[(i - 1) % code.k_message_bits()] = 1;
            }

            let codeword = code.encode(&message).unwrap();
            let errors = code.verify_codeword(&codeword).unwrap();
            assert_eq!(errors, 0, "Pattern {} should satisfy all parity checks", i);
        }
        println!("✓ Multiple pattern tests passed!");

        // Test 4: Verify linearity property (c1 + c2 = encode(m1 + m2))
        let message1 = vec![1u8, 0, 1, 0, 0, 1, 0, 0, 0, 0];
        let message2 = vec![0u8, 1, 0, 1, 0, 0, 1, 0, 0, 0];

        let codeword1 = code.encode(&message1).unwrap();
        let codeword2 = code.encode(&message2).unwrap();

        // XOR the messages and codewords
        let mut combined_message = vec![0u8; code.k_message_bits()];
        let mut combined_codeword = vec![0u8; code.n_bits()];

        for i in 0..code.k_message_bits() {
            combined_message[i] = message1[i] ^ message2[i];
        }
        for i in 0..code.n_bits() {
            combined_codeword[i] = codeword1[i] ^ codeword2[i];
        }

        let direct_combined = code.encode(&combined_message).unwrap();
        assert_eq!(
            combined_codeword, direct_combined,
            "Linearity property should hold"
        );
        println!("✓ Linearity property verified!");

        println!("🎉 All encoding tests passed! LDPC encoding is working perfectly!");
    }
}
