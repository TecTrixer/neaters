use serde::{Deserialize, Serialize};

use crate::neuralnetwork::NeuralNetwork;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Species {
    pub representative: NeuralNetwork,
    pub members: Vec<usize>,
}

impl Species {
    pub fn new_with_network(nn: NeuralNetwork) -> Self {
        Species {
            members: vec![nn.id],
            representative: nn,
        }
    }
    pub fn clear(&mut self) {
        self.members.clear();
    }

    pub fn is_unused(&self) -> bool {
        self.members.is_empty()
    }
}
