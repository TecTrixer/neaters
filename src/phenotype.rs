use crate::neuralnetwork::Node;
use crate::neuralnetwork::NodeType;
use crate::NeuralNetwork;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;

#[derive(Debug, PartialEq)]
pub struct Phenotype {
    edges: Vec<Vec<(usize, f32)>>,
    pub node_value_array: Vec<f32>,
    pub topo_order: Vec<usize>,
    outputs: Vec<usize>,
}

impl Phenotype {
    pub fn from_nn(nn: &NeuralNetwork) -> Self {
        let input_length = nn.size.0;
        let output_length = nn.size.1;
        let node_index_map = Phenotype::create_node_index_mapping(&nn.nodes);
        let mut edges: Vec<Vec<(usize, f32)>> = Vec::with_capacity(nn.nodes.len());
        for _ in 0..nn.nodes.len() {
            edges.push(Vec::new());
        }
        let mut outputs: Vec<usize> = Vec::with_capacity(output_length);
        for node in nn.nodes.iter() {
            match node.node_type {
                NodeType::Hidden => break,
                NodeType::Output => outputs.push(*node_index_map.get(&node.id).unwrap()),
                NodeType::Input => (),
            }
        }
        for edge in nn.edges.iter() {
            let from = *node_index_map.get(&edge.from).unwrap();
            let to = *node_index_map.get(&edge.to).unwrap();
            if edge.enabled {
                edges[from].push((to, edge.weight));
            }
        }
        let node_value_array: Vec<f32> = Vec::with_capacity(nn.nodes.len());
        let topo_order: Vec<usize> = Phenotype::create_topo_order(&edges, input_length);
        Phenotype {
            edges,
            node_value_array,
            topo_order,
            outputs,
        }
    }

    fn create_topo_order(edges: &[Vec<(usize, f32)>], input_nodes: usize) -> Vec<usize> {
        let mut stack: Vec<usize> = Vec::new();
        // add every input node (one more than input bc of the constant) to the stack for dfs
        for i in 0..=input_nodes {
            stack.push(i);
        }
        let mut visited: Vec<bool> = Vec::with_capacity(edges.len());
        let mut order: Vec<usize> = Vec::with_capacity(edges.len());
        for _ in 0..edges.len() {
            visited.push(false);
            order.push(0);
        }
        let mut idx: usize = edges.len() - 1;
        while !stack.is_empty() {
            let elem = stack[stack.len() - 1];
            if visited[elem] {
                stack.pop();
                order[idx] = elem;
                if idx > 0 {
                    idx -= 1;
                }
                continue;
            } else {
                visited[elem] = true;
            }
            for edge in edges[elem].iter() {
                let to = edge.0;
                if !visited[to] {
                    stack.push(to);
                }
            }
        }
        order
    }

    fn create_node_index_mapping(nodes: &[Node]) -> FxHashMap<usize, usize> {
        let mut map: FxHashMap<usize, usize> = FxHashMap::with_capacity_and_hasher(
            nodes.len(),
            BuildHasherDefault::<FxHasher>::default(),
        );
        for (idx, node) in nodes.iter().enumerate() {
            map.insert(idx, node.id);
        }
        map
    }

    pub fn compute(&mut self, inputs: Vec<f32>) -> Vec<f32> {
        let mut outputs: Vec<f32> = Vec::with_capacity(self.outputs.len());
        self.node_value_array.push(1.0);
        let input_length = inputs.len();
        for input in inputs.into_iter() {
            self.node_value_array.push(input);
        }
        for _ in (input_length + 1)..self.edges.len() {
            self.node_value_array.push(0.0);
        }
        for node in self.topo_order.iter() {
            self.node_value_array[*node] = sigmoid(self.node_value_array[*node]);
            for (to, weight) in self.edges[*node].iter() {
                self.node_value_array[*to] += *weight * self.node_value_array[*node];
            }
        }
        for o_idx in self.outputs.iter() {
            outputs.push(self.node_value_array[*o_idx]);
        }
        outputs
    }

    pub fn reset(&mut self) {
        self.node_value_array.clear();
    }
}

// we can also choose another activation function
pub fn sigmoid(x: f32) -> f32 {
    x / (1.0 + x.abs())
}
