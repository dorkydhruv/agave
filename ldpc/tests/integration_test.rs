use ::ldpc::*;

#[test]
fn test_end_to_end_code_creation() {
    // Test the main use case
    let code = LdpcCode::new(
        100,                  // 100 parity checks
        200,                  // 200 total bits
        12345,                // Random seed
        MakeMethod::EvenBoth, // Balanced construction
        "0.2x3/0.8x6",        // Mixed degree distribution
        true,                 // Eliminate 4-cycles
    )
    .expect("Failed to create LDPC code");

    // Verify basic properties
    assert_eq!(code.n_bits(), 200);
    assert_eq!(code.m_checks(), 100);
    assert_eq!(code.k_message_bits(), 100);
    assert!((code.rate() - 0.5).abs() < 1e-10);

    // Test that zero codeword is valid
    let zero_codeword = vec![0u8; 200];
    let errors = code.verify_codeword(&zero_codeword).unwrap();
    assert_eq!(errors, 0);

    println!(
        "Successfully created LDPC({}, {}) code with rate {:.3}",
        code.n_bits(),
        code.k_message_bits(),
        code.rate()
    );
}

#[test]
fn test_different_constructions() {
    let methods = [MakeMethod::EvenCol, MakeMethod::EvenBoth];
    let distributions = ["3", "4", "0.5x3/0.5x5"];

    for &method in &methods {
        for &dist in &distributions {
            let code = LdpcCode::new(50, 100, 1, method, dist, false)
                .expect(&format!("Failed with method {:?}, dist {}", method, dist));

            assert!(code.parity_check_matrix().is_some());

            // Verify zero codeword
            let errors = code.verify_codeword(&vec![0u8; 100]).unwrap();
            assert_eq!(errors, 0);
        }
    }
}

#[test]
fn test_classic_ldpc_examples() {
    // Test some classic LDPC code parameters

    // Regular (3,6) LDPC code
    let regular_code = LdpcCode::new(1000, 2000, 42, MakeMethod::EvenBoth, "3", false).unwrap();
    assert_eq!(regular_code.rate(), 0.5);

    // Irregular LDPC code similar to those used in practice
    let irregular_code = LdpcCode::new(
        800,
        1600,
        123,
        MakeMethod::EvenBoth,
        "0.1x2/0.4x3/0.5x6",
        false,
    )
    .unwrap();
    assert_eq!(irregular_code.rate(), 0.5);

    // High rate code
    let high_rate_code =
        LdpcCode::new(200, 1000, 999, MakeMethod::EvenBoth, "0.7x3/0.3x4", false).unwrap();
    assert_eq!(high_rate_code.rate(), 0.8);

    // All should have valid zero codewords
    for code in [&regular_code, &irregular_code, &high_rate_code] {
        let zero_codeword = vec![0u8; code.n_bits()];
        let errors = code.verify_codeword(&zero_codeword).unwrap();
        assert_eq!(errors, 0);
    }
}
