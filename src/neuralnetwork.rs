use bincode;
use serde::{Deserialize, Serialize};
use std::fs::OpenOptions;
use std::io::{BufReader, Read, Write};

#[derive(Debug, PartialEq, Deserialize, Serialize)]
struct Node {
    id: usize,
    node_type: NodeType,
}

impl Node {
    fn input_with_id(id: usize) -> Self {
        return Node {
            id,
            node_type: NodeType::Input,
        };
    }
    fn output_with_id(id: usize) -> Self {
        return Node {
            id,
            node_type: NodeType::Output,
        };
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
enum NodeType {
    Input,
    Hidden,
    Output,
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
struct Edge {
    from: usize,
    to: usize,
    weight: f32,
    enabled: bool,
    innovation: usize,
}

impl Edge {
    fn initial_from_to(from: usize, to: usize) -> Self {
        return Edge {
            from,
            to,
            weight: 1.0,
            enabled: true,
            innovation: 0,
        };
    }
}

#[derive(Debug, Deserialize, PartialEq, Serialize)]
pub struct NeuralNetwork {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl NeuralNetwork {
    pub fn new(input_nodes: usize, output_nodes: usize) -> Self {
        // TODO: sanitize input (output_nodes = 0?)
        let mut nodes = vec![];
        let mut edges = vec![];
        // creating input nodes + edges
        // it also creates a default input node with its input as a constant 1.0
        for i in 0..=input_nodes {
            // add input node
            nodes.push(Node::input_with_id(i));
            // for this input node add a default edge with weight 1.0 to every output node
            for j in (input_nodes + 1)..=(input_nodes + output_nodes) {
                edges.push(Edge::initial_from_to(i, j));
            }
        }
        // add output nodes
        for i in (input_nodes + 1)..=(input_nodes + output_nodes) {
            nodes.push(Node::output_with_id(i));
        }
        return NeuralNetwork { nodes, edges };
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
        let bytes = NeuralNetwork::load_bytes_from(at);
        return NeuralNetwork::create_from_bytes(bytes);
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
