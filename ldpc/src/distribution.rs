use thiserror::Error;

#[derive(Error, Debug)]
pub enum DistributionError {
    #[error("Invalid distribution format")]
    InvalidFormat,
    #[error("Invalid proportion: {0}")]
    InvalidProportion(String),
    #[error("Invalid number: {0}")]
    InvalidNumber(String),
    #[error("Empty distribution")]
    Empty,
}

#[derive(Debug, Clone)]
struct DistribEntry {
    prop: f64,
    num: usize,
}

#[derive(Debug, Clone)]
pub struct Distribution {
    size: usize,
    list: Vec<DistribEntry>,
}

impl Distribution {
    /// CREATE A DISTRIBUTION - exact port of distrib_create algorithm
    pub fn create(c: &str) -> Result<Self, DistributionError> {
        // Handle single number case first
        if let Ok(n) = c.parse::<usize>() {
            if n > 0 {
                return Ok(Distribution {
                    size: 1,
                    list: vec![DistribEntry { prop: 1.0, num: n }],
                });
            }
        }

        // Parse format like "0.5x2/0.3x3/0.2x4"
        let parts: Vec<&str> = c.split('/').collect();
        if parts.is_empty() {
            return Err(DistributionError::Empty);
        }

        let mut entries = Vec::new();
        let mut sum = 0.0;

        for part in parts {
            let components: Vec<&str> = part.split('x').collect();
            if components.len() != 2 {
                return Err(DistributionError::InvalidFormat);
            }

            let prop: f64 = components[0]
                .parse()
                .map_err(|_| DistributionError::InvalidProportion(components[0].to_string()))?;
            let num: usize = components[1]
                .parse()
                .map_err(|_| DistributionError::InvalidNumber(components[1].to_string()))?;

            if prop <= 0.0 || num == 0 {
                return Err(DistributionError::InvalidFormat);
            }

            entries.push(DistribEntry { prop, num });
            sum += prop;
        }

        // Normalize proportions
        for entry in &mut entries {
            entry.prop /= sum;
        }

        Ok(Distribution {
            size: entries.len(),
            list: entries,
        })
    }

    /// RETURN THE MAXIMUM NUMBER - exact port of distrib_max
    pub fn max(&self) -> usize {
        self.list.iter().map(|e| e.num).max().unwrap_or(0)
    }

    pub fn size(&self) -> usize {
        self.size
    }
    pub fn prop(&self, i: usize) -> f64 {
        self.list[i].prop
    }
    pub fn num(&self, i: usize) -> usize {
        self.list[i].num
    }
}
