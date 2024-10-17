use serde::{Deserialize, Serialize};

/// Configuration for training, all sorts of parameters are stored in here
#[derive(Copy, Deserialize, Debug, Serialize, Clone, PartialEq)]
pub struct Config {
    /// Determines how much the number of excess genes influences the compatibility distance
    pub c1: f32,
    /// Determines how much the number of disjoint genes influences the compatibility distance
    pub c2: f32,
    /// Determines how much the average weight difference of matching edges influences the compatibility distance
    pub c3: f32,
}

// TODO: find useful default parameters
impl Default for Config {
    fn default() -> Self {
        Config {
            c1: 1.0,
            c2: 1.0,
            c3: 1.0,
        }
    }
}
