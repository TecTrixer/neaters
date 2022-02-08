use crate::neuralnetwork::NeuralNetwork;
use bincode;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufReader, Read, Write};

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Solver {
    networks: Vec<NeuralNetwork>,
    network_size: (usize, usize),
    generation_size: usize,
    fitness: Vec<f32>,
    generation: usize,
}

impl Solver {
    // TODO: sanitize input (no generation size of 0, ...)
    pub fn with_size(input_nodes: usize, output_nodes: usize, generation_size: usize) -> Self {
        let mut networks: Vec<NeuralNetwork> = Vec::with_capacity(generation_size);
        for i in 0..generation_size {
            networks.push(NeuralNetwork::with_size_and_id(
                input_nodes,
                output_nodes,
                i,
            ));
        }
        let fitness: Vec<f32> = Vec::with_capacity(generation_size);
        return Solver {
            networks,
            network_size: (input_nodes, output_nodes),
            generation_size,
            fitness,
            generation: 0,
        };
    }

    // NOTE: should this be public?
    pub fn as_byte_representation(&self) -> Vec<u8> {
        // encode neural network as binary
        let encoded: Vec<u8> = match bincode::serialize(&self) {
            Ok(bytes) => bytes,
            // TODO: add clean error handling
            Err(_) => vec![],
        };
        return encoded;
    }

    pub fn save_as(&self, at: &str) {
        let encoded = self.as_byte_representation();
        // TODO: handle errors
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .open(at)
            .unwrap();
        file.write_all(&encoded).unwrap();
    }

    pub fn load_from(at: &str) -> Self {
        let bytes = Solver::load_bytes_from(at);
        return Solver::create_from_bytes(bytes);
    }

    // NOTE: should this be public?
    // NOTE: should we use BufReader or just a normal read from a File?
    pub fn load_bytes_from(at: &str) -> Vec<u8> {
        // TODO: handle io errors
        let file = OpenOptions::new().read(true).open(at).unwrap();
        let mut buf_reader = BufReader::new(file);
        let mut buffer: Vec<u8> = Vec::new();
        buf_reader.read_to_end(&mut buffer).unwrap();
        return buffer;
    }

    fn create_from_bytes(bytes: Vec<u8>) -> Self {
        // TODO: handle serialization errors
        let decoded: Self = bincode::deserialize(&bytes).unwrap();
        return decoded;
    }
}
