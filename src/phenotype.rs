use crate::neuralnetwork::Node;
use crate::neuralnetwork::NodeType;
use crate::NeuralNetwork;
use rustc_hash::{FxHashMap, FxHasher};
use std::hash::BuildHasherDefault;

/// Graph representation of NeuralNetwork, used to compute its output.
#[derive(Debug, PartialEq, Clone)]
pub struct Phenotype {
    /// EdgeList with the destination and the weight of each edge for each node.
    edges: Vec<Vec<(usize, f32)>>,
    /// Array used to store and mutate the values of each node.
    pub node_value_array: Vec<f32>,
    /// Order in which the nodes need to be processed such that all predecessors of a node have
    /// been processed before it is being processed itself.
    pub topo_order: Vec<usize>,
    /// List of indexes of the outputs of the network.
    outputs: Vec<usize>,
}

impl Phenotype {
    /// Construct a Phenotype from a NeuralNetwork.
    /// At first create an index mapping so that you know which NodeId you need to map to which
    /// index. For this a fast FxHashMap is being used.
    ///
    /// Afterwards initialize an empty EdgeList and construct the outputs index array.
    /// Then populating the EdgeList with the correct values using the index mapping.
    /// At last creating the node value array and computing the topological order.
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

    /// Function to create the topological order for computation of the network without any
    /// uncomputed predecessors. Using DFS to create the order.
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

    /// Creating the node index mapping using a simple and very fast hashmap.
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

    /// Computing the output of the network depending on the input values.
    ///
    /// At first filling the node values, then traversing the network in topological order.
    /// For each node in the beginning calculate the sigmoid value of its own value and then for
    /// each edge of that node add the edge weight times the node's value to the destination node.
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

    /// Reset the phenotype for reused computation
    pub fn reset(&mut self) {
        self.node_value_array.clear();
    }
}

// we can also choose another activation function
/// Sigmoid activation function
/// Approximated by x/(1+|x|)
pub fn sigmoid(x: f32) -> f32 {
    x / (1.0 + x.abs())
}
