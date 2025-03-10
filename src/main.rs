use quantum_mesh::quantum_crypto::{QuantumCrypto, QuantumCryptoError, HasCoordinates, HasTransactionData};
use std::collections::HashMap;
use sha2::{Sha256, Digest};
use libp2p::{
    gossipsub::{
        self,
        Behaviour as GossipsubBehaviour,
        Config as GossipsubConfig,
        MessageAuthenticity,
    },
    identity, mdns,
    swarm::{NetworkBehaviour, SwarmEvent},
    SwarmBuilder,
    noise, tcp, yamux, PeerId,
    futures::StreamExt,
    Swarm,
};
use rand;
use std::{error::Error, time::{SystemTime, UNIX_EPOCH}};
use serde::{Serialize, Deserialize};
use std::time::Duration;
use std::collections::HashSet;
use std::collections::VecDeque;
use lru::LruCache;
use std::sync::Mutex;
use std::sync::Arc;
use bincode;
use quantum_mesh::geometry::{Point, hyperbolic_distance};



#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Transaction {
    pub from: String,
    pub to: String,
    pub amount: f64,
    pub timestamp: u64,
    pub signature: String,
}



impl HasTransactionData for Transaction {
    fn get_sender(&self) -> &str { &self.from }
    fn get_receiver(&self) -> &str { &self.to }
    fn get_amount(&self) -> f64 { self.amount }
}

#[derive(Clone, Debug)]
struct User {
    pub id: String,
    pub keypair: identity::Keypair,
    pub node: Node,
    pub serialized_keypair: Option<SerializableKeypair>,
    pub quantum_crypto: Arc<QuantumCrypto>,  // Add this field
}

impl User {
    fn new(id: String, keypair: identity::Keypair, node: Node) -> Result<Self, QuantumCryptoError> {
        // Initialize quantum-resistant cryptography
        let quantum_crypto = Arc::new(QuantumCrypto::new()?);
        
        Ok(User {
            id,
            keypair,
            node,
            serialized_keypair: None,
            quantum_crypto,
        })
    }
}

impl Serialize for User {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("User", 3)?;
        state.serialize_field("id", &self.id)?;
        state.serialize_field("node", &self.node)?;
        
        let keypair_bytes = self.keypair.to_protobuf_encoding()
            .map_err(serde::ser::Error::custom)?;
        state.serialize_field("keypair_bytes", &keypair_bytes)?;
        
        state.end()
    }
}

impl<'de> Deserialize<'de> for User {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct UserHelper {
            id: String,
            node: Node,
            keypair_bytes: Vec<u8>,
        }

        let helper = UserHelper::deserialize(deserializer)?;
        let keypair = identity::Keypair::from_protobuf_encoding(&helper.keypair_bytes)
            .map_err(serde::de::Error::custom)?;

        // Map the QuantumCryptoError to a deserialization error
        let quantum_crypto = Arc::new(QuantumCrypto::new()
            .map_err(|e| serde::de::Error::custom(e.to_string()))?);

        Ok(User {
            id: helper.id,
            keypair,
            node: helper.node,
            serialized_keypair: None,
            quantum_crypto,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Node {
    pub position: Point,
    pub connections: Vec<String>,
    pub transactions: Vec<Transaction>,
    pub balance: f64,
    pub is_validator: bool,  // Add this field
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Block {
    transactions: Vec<Transaction>,
    timestamp: u64,
    previous_hash: String,
    hash: String,
    difficulty: u32,
    nonce: u64,
    merkle_root: String,
}

#[derive(Clone, Debug)]
struct MerkleTree {
    root: String,
    leaves: Vec<String>,
}

#[derive(Debug, Clone)]
struct HyperbolicRegion {
    center: Point,
    radius: f64,
    nodes: Vec<String>, // Node IDs in this region
    boundary_points: Vec<Point>,
}

#[derive(Clone, Debug)]
struct Shard {
    id: usize,
    nodes: Vec<String>, // Node IDs in this shard
    transactions: Vec<Transaction>,
    center: Point,
    radius: f64,
}

impl Shard {
    fn new(id: usize, center: Point, radius: f64) -> Self {
        Shard {
            id,
            nodes: Vec::new(),
            transactions: Vec::new(),
            center,
            radius,
        }
    }

    fn calculate_metrics_from_users(&self, users: &HashMap<String, User>) -> RegionMetrics {
        let mut total_transactions = 0;
        let mut total_load = 0.0;

        for node_id in &self.nodes {
            if let Some(user) = users.get(node_id) {
                total_transactions += user.node.transactions.len();
                // Calculate load based on transaction history and connections
                let node_load = user.node.transactions.len() as f64 * 
                               user.node.connections.len() as f64;
                total_load += node_load;
            }
        }

        RegionMetrics {
            transaction_density: total_transactions as f64 / (std::f64::consts::PI * self.radius * self.radius),
            node_count: self.nodes.len(),
            average_load: if self.nodes.is_empty() { 0.0 } else { total_load / self.nodes.len() as f64 },
            boundary_crossings: 0,
        }
    }
}

#[derive(Debug)]
struct RoutingMetrics {
    path_hits: u64,
    path_misses: u64,
    average_path_length: f64,
    routing_time_ms: u64,
}

struct HyperbolicRouter {
    regions: Vec<HyperbolicRegion>,
    routing_table: HashMap<String, Vec<String>>, // NodeId -> [Neighbor NodeIds]
    path_cache: HashMap<(String, String), Vec<Point>>, // (From, To) -> Path
    metrics: RoutingMetrics,
    max_cache_size: usize,
    shards: Vec<Shard>,
}



impl HyperbolicRouter {
    fn new(max_cache_size: usize) -> Self {
        HyperbolicRouter {
            regions: Vec::new(),
            routing_table: HashMap::new(),
            path_cache: HashMap::new(),
            metrics: RoutingMetrics {
                path_hits: 0,
                path_misses: 0,
                average_path_length: 0.0,
                routing_time_ms: 0,
            },
            max_cache_size,
            shards: Vec::new(),
        }
    }

    fn calculate_geodesic_path(&self, from: &Point, to: &Point) -> Vec<Point> {
        // Calculate path in hyperbolic space using Poincaré disk model
        let steps = 10; // Number of points along the path
        let mut path = Vec::with_capacity(steps);
        
        for i in 0..steps {
            let t = i as f64 / (steps - 1) as f64;
            // Linear interpolation in hyperbolic space
            let x = from.x * (1.0 - t) + to.x * t;
            let y = from.y * (1.0 - t) + to.y * t;
            // Project back onto Poincaré disk
            let norm = (x * x + y * y).sqrt();
            if norm > 1.0 {
                let scale = 0.99 / norm; // Stay within unit disk
                path.push(Point { x: x * scale, y: y * scale });
            } else {
                path.push(Point { x, y });
            }
        }
        
        path
    }

    fn update_routing_table(&mut self, users: &HashMap<String, User>) {
        for (node_id, node_user) in users {
            let mut neighbors = Vec::new();
            let node_pos = &node_user.node.position;
            
            // Find closest nodes in hyperbolic space
            for (other_id, other_user) in users {
                if other_id != node_id {
                    // Calculate hyperbolic distance directly here since we don't have mesh
                    let dx = other_user.node.position.x - node_pos.x;
                    let dy = other_user.node.position.y - node_pos.y;
                    let euclidean_dist = (dx * dx + dy * dy).sqrt();
                    let distance = 2.0 * ((1.0 + euclidean_dist) / (1.0 - euclidean_dist)).ln();
                    
                    if distance < 2.0 { // Threshold for considering nodes as neighbors
                        neighbors.push(other_id.clone());
                    }
                }
            }
            
            self.routing_table.insert(node_id.clone(), neighbors);
        }
    }

    fn optimize_regions(&mut self, users: &HashMap<String, User>) {
        // Calculate metrics for all regions
        let region_metrics: Vec<RegionMetrics> = self.regions
            .iter()
            .map(|region| region.calculate_metrics_from_users(users))
            .collect();
        
        // Check if regions need to be split or merged
        self.handle_region_splits(users, &region_metrics);
        self.handle_region_merges(users, &region_metrics);
        
        // Adjust region boundaries based on load
        self.adjust_region_boundaries(&region_metrics);
    }

    fn handle_region_splits(&mut self, users: &HashMap<String, User>, metrics: &[RegionMetrics]) {
        let mut new_regions = Vec::new();
        
        for (i, region) in self.regions.clone().iter().enumerate() {
            if metrics[i].transaction_density > 1000.0 || metrics[i].node_count > 100 {
                // Split region into two
                let (region1, region2) = self.split_region(region, users);
                new_regions.push(region1);
                new_regions.push(region2);
            } else {
                new_regions.push(region.clone());
            }
        }
        
        self.regions = new_regions;
    }

    fn split_region(&self, region: &HyperbolicRegion, users: &HashMap<String, User>) -> (HyperbolicRegion, HyperbolicRegion) {
        // Find the axis of maximum spread
        let mut nodes: Vec<_> = region.nodes.iter()
            .filter_map(|id| users.get(id))
            .collect();
        
        // Sort nodes by x coordinate
        nodes.sort_by(|a, b| a.node.position.x.partial_cmp(&b.node.position.x).unwrap());
        
        let mid = nodes.len() / 2;
        let (left_nodes, right_nodes): (Vec<_>, Vec<_>) = nodes.iter()
            .map(|user| user.id.clone())
            .partition(|id| {
                let user = users.get(id).unwrap();
                user.node.position.x < nodes[mid].node.position.x
            });
        
        // Create two new regions
        let (center1, radius1) = self.calculate_region_parameters(&left_nodes, users);
        let (center2, radius2) = self.calculate_region_parameters(&right_nodes, users);
        
        (
            HyperbolicRegion::new(center1, radius1),
            HyperbolicRegion::new(center2, radius2)
        )
    }

    fn handle_region_merges(&mut self, users: &HashMap<String, User>, metrics: &[RegionMetrics]) {
        let mut i = 0;
        while i < self.regions.len() {
            if metrics[i].transaction_density < 100.0 && metrics[i].node_count < 10 {
                // Find closest region to merge with
                if let Some((j, _)) = self.find_closest_region(i, users) {
                    self.merge_regions(i, j, users);
                    continue;
                }
            }
            i += 1;
        }
    }

    fn find_closest_region(&self, region_idx: usize, _users: &HashMap<String, User>) -> Option<(usize, f64)> {
        let region = &self.regions[region_idx];
        let mut closest = None;
        let mut min_distance = f64::MAX;
        
        for (i, other) in self.regions.iter().enumerate() {
            if i != region_idx {
                let distance = hyperbolic_distance(&region.center, &other.center);
                if distance < min_distance {
                    min_distance = distance;
                    closest = Some((i, distance));
                }
            }
        }
        
        closest
    }

    fn merge_regions(&mut self, i: usize, j: usize, users: &HashMap<String, User>) {
        let region1 = self.regions[i].clone();
        let region2 = self.regions[j].clone();
        
        // Combine node lists
        let mut combined_nodes = region1.nodes.clone();
        combined_nodes.extend(region2.nodes.clone());
        
        // Calculate new region parameters
        let (center, radius) = self.calculate_region_parameters(&combined_nodes, users);
        let mut new_region = HyperbolicRegion::new(center, radius);
        new_region.nodes = combined_nodes;
        
        // Remove old regions and add new one
        let max_idx = i.max(j);
        let min_idx = i.min(j);
        self.regions.remove(max_idx);
        self.regions.remove(min_idx);
        self.regions.push(new_region);
    }

    fn adjust_region_boundaries(&mut self, metrics: &[RegionMetrics]) {
        for (i, region) in self.regions.iter_mut().enumerate() {
            let density = metrics[i].transaction_density;
            let load = metrics[i].average_load;
            
            // Adjust radius based on load and density
            if density > 500.0 || load > 1000.0 {
                region.radius *= 0.9; // Shrink overloaded regions
            } else if density < 100.0 || load < 100.0 {
                region.radius *= 1.1; // Expand underutilized regions
            }
            
            // Update boundary points
            region.boundary_points = HyperbolicRegion::calculate_boundary_points(&region.center, region.radius);
        }
    }

    fn assign_nodes_to_shards(&mut self, users: &HashMap<String, User>) {
        // Clear existing assignments
        for shard in &mut self.shards {
            shard.nodes.clear();
        }

        // Assign each user to the closest shard
        for (user_id, user) in users {
            let mut min_distance = f64::MAX;
            let mut closest_shard_index = 0;

            // Find the closest shard
            for (i, shard) in self.shards.iter().enumerate() {
                let distance = self.hyperbolic_distance(&user.node.position, &shard.center);
                if distance < min_distance {
                    min_distance = distance;
                    closest_shard_index = i;
                }
            }

            // Assign user to the closest shard
            if let Some(shard) = self.shards.get_mut(closest_shard_index) {
                shard.nodes.push(user_id.clone());
            }
        }
    }

    fn adjust_shard_boundaries(&mut self, users: &HashMap<String, User>) {
        // First collect the metrics
        let metrics: Vec<_> = self.shards.iter()
            .map(|shard| shard.calculate_metrics_from_users(users))
            .collect();
        
        // Calculate new centers first
        let new_centers: Vec<Point> = self.shards.iter()
            .map(|shard| self.calculate_shard_center(&shard.nodes, users))
            .collect();
        
        // Then update the shards based on metrics
        for i in 0..self.shards.len() {
            // Get a reference to the current shard
            if let Some(shard) = self.shards.get_mut(i) {
                // Adjust radius based on transaction density
                if metrics[i].transaction_density > 1000.0 {
                    shard.radius *= 0.9; // Shrink overloaded shards
                } else if metrics[i].transaction_density < 100.0 {
                    shard.radius *= 1.1; // Expand underutilized shards
                }

                // Update shard center by accessing new_centers by reference
                if let Some(new_center) = new_centers.get(i) {
                    shard.center.x = new_center.x;
                    shard.center.y = new_center.y;
                }
            }
        }
    }

    fn calculate_shard_center(&self, nodes: &[String], users: &HashMap<String, User>) -> Point {
        // Calculate the center of the shard based on node positions
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let count = nodes.len() as f64;

        for node_id in nodes {
            if let Some(user) = users.get(node_id) {
                x_sum += user.node.position.x;
                y_sum += user.node.position.y;
            }
        }

        Point {
            x: x_sum / count,
            y: y_sum / count,
        }
    }

    fn hyperbolic_distance(&self, p1: &Point, p2: &Point) -> f64 {
        hyperbolic_distance(p1, p2)
    }

    fn find_shard_for_node(&self, node_id: &str) -> usize {
        // Find the shard containing the node
        for (i, shard) in self.shards.iter().enumerate() {
            if shard.nodes.contains(&node_id.to_string()) {
                return i;
            }
        }
        0 // Return default shard if not found
    }

    fn find_closest_node(&self, point: &Point) -> Option<(String, f64)> {
        self.routing_table.keys()
            .map(|id| (id.clone(), self.hyperbolic_distance(point, &Point { x: 0.0, y: 0.0 }))) // You'll need to get actual node positions
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    fn initialize_regions(&mut self, users: &HashMap<String, User>) {
        // Clear existing shards
        self.shards.clear();

        // Create at least one initial shard if there are no users
        if users.is_empty() {
            self.shards.push(Shard::new(0, Point { x: 0.0, y: 0.0 }, 1.0));
            return;
        }

        // Calculate center of mass for initial shard
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        let user_count = users.len() as f64;

        for user in users.values() {
            center_x += user.node.position.x;
            center_y += user.node.position.y;
        }

        center_x /= user_count;
        center_y /= user_count;

        // Create initial shard
        self.shards.push(Shard::new(
            0,
            Point { x: center_x, y: center_y },
            1.0  // Initial radius
        ));
    }
}

struct QuantumMesh {
    users: HashMap<String, User>,
    pending_transactions: VecDeque<Transaction>, // Change from Vec to VecDeque
    blocks: Vec<Block>,
    difficulty: u32,
    metrics: NetworkMetrics,
    pending_messages: HashMap<String, (MeshMessage, u32)>,
    sync_state: HashMap<String, usize>,
    swarm: Option<Swarm<MeshBehaviour>>,
    local_peer_id: Option<PeerId>,
    router: HyperbolicRouter,
    compressed_state_cache: Mutex<LruCache<String, CompressedState>>,
}

#[derive(NetworkBehaviour)]
#[behaviour(to_swarm = "MeshBehaviourEvent")]
struct MeshBehaviour {
    gossipsub: GossipsubBehaviour,
    mdns: mdns::async_io::Behaviour,
}

#[derive(Debug)]
enum MeshBehaviourEvent {
    Gossipsub(gossipsub::Event),
    Mdns(mdns::Event),
}

impl From<gossipsub::Event> for MeshBehaviourEvent {
    fn from(event: gossipsub::Event) -> Self {
        MeshBehaviourEvent::Gossipsub(event)
    }
}

impl From<mdns::Event> for MeshBehaviourEvent {
    fn from(event: mdns::Event) -> Self {
        MeshBehaviourEvent::Mdns(event)
    }
}

#[derive(Debug, Serialize, Deserialize, Clone)]
enum MeshMessage {
    Transaction(Transaction),
    Block(Block),
    RequestChain { from_block: usize },
    ChainResponse { blocks: Vec<Block> },
    MessageAck { message_id: String },
    RetransmissionRequest { message_id: String },
    SyncStatus { height: usize, peer_id: String },
    EncryptedTransaction {
        ciphertext: Vec<u8>,
        sender_id: String,
    },
    QuantumKeyExchange {
        public_key: Vec<u8>,
        sender_id: String,
    },
}

#[derive(Default)]
struct NetworkMetrics {
    messages_sent: u64,
    messages_received: u64,
    failed_validations: u64,
    retransmissions: u64,
    sync_requests: u64,
    pub states_compressed: u64,
    pub total_original_size: u64,
    pub total_compressed_size: u64,
    pub total_compression_time: u64,
    pub compression_failures: u64,
}

#[derive(Debug)]
enum MetricEvent {
    MessageSent,
    MessageReceived,
    ValidationFailed,
    SyncRequested,
}

#[derive(Debug, Clone)]
struct RegionMetrics {
    transaction_density: f64,
    node_count: usize,
    average_load: f64,
    boundary_crossings: u64,
}

// Add these new structures after the existing ones
#[derive(Debug, Clone)]
struct CompressedState {
    geodesic_path: Vec<Point>,
    compressed_data: Vec<u8>,
    compression_ratio: f64,
    proof: GeometricProof,
}

// Update the GeometricProof structure to include all required fields
#[derive(Debug, Clone)]
struct GeometricProof {
    tessellation_points: Vec<Point>,
    verification_path: Vec<Point>,
    position_data: Vec<u8>,  // Add this field
    hash: String,           // Add this field
    signature: Vec<u8>,     // Add this field
    timestamp: u64,         // Add this field
}

// Add these new structures after the existing CompressedState and GeometricProof

#[derive(Debug)]
struct ReconstructedState {
    nodes: Vec<Point>,
    confidence: f64,
    verification_hash: String,
}

// Define a type alias for the cache
type CompressedStateCache = LruCache<String, CompressedState>;

impl QuantumMesh {
    fn new(difficulty: u32, cache_capacity: usize) -> Self {
        let mut mesh = QuantumMesh {
            users: HashMap::new(),
            pending_transactions: VecDeque::new(), // Initialize as VecDeque
            blocks: Vec::new(),
            difficulty,
            metrics: NetworkMetrics::default(),
            pending_messages: HashMap::new(),
            sync_state: HashMap::new(),
            swarm: None,
            local_peer_id: None,
            router: HyperbolicRouter::new(1000),
            compressed_state_cache: Mutex::new(LruCache::new(cache_capacity)),
        };
        
        // Initialize regions (will be empty at first since no users)
        mesh.router.initialize_regions(&mesh.users);
        mesh
    }

    fn add_node(&mut self, id: String, x: f64, y: f64, initial_balance: f64) -> Result<(), QuantumCryptoError> {
        let node = Node {
            position: Point { x, y },
            connections: Vec::new(),
            transactions: Vec::new(),
            balance: initial_balance,
            is_validator: false,  // Default to false
        };
        
        let user = User::new(
            id.clone(),
            identity::Keypair::generate_ed25519(),
            node,
        )?;
        
        self.users.insert(id, user);
        self.router.optimize_regions(&self.users);
        Ok(())
    }

    fn hyperbolic_distance(&self, p1: &Point, p2: &Point) -> f64 {
        hyperbolic_distance(p1, p2)
    }

    // Update the validate_transaction method to handle async correctly
    async fn validate_transaction(&self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        // Basic validation
        if !self.validate_transaction_basics(transaction) {
            return Ok(false);
        }

        // Get sender for proof verification
        let sender = self.users.get(&transaction.from)
            .ok_or("Sender not found")?;
        
        // Serialize transaction for proof verification
        let transaction_data = bincode::serialize(&transaction)?;
        
        // Generate and verify the proof
        let position = sender.node.position;
        let proof = sender.quantum_crypto.generate_geometric_proof(&transaction_data, &position)?;
        
        Ok(sender.quantum_crypto.verify_geometric_proof(&proof, &transaction_data)?)
    }

    fn add_transaction(&mut self, from: String, to: String, amount: f64) -> bool {
        let transaction = Transaction {
            from: from.clone(),
            to: to.clone(),
            amount,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: format!("{}_{}_{}_{}", from, to, amount, SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs()),
        };

        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                if self.validate_transaction(&transaction).await? {
                    self.pending_transactions.push_back(transaction);
                    Ok::<bool, Box<dyn Error>>(true)
                } else {
                    Ok::<bool, Box<dyn Error>>(false)
                }
            })
            .unwrap_or(false)
    }

    fn mine_block(&self, transactions: &[Transaction], previous_hash: &str) -> Block {
        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        
        let merkle_tree = MerkleTree::new(transactions);
        let mut nonce = 0u64;
        
        loop {
            let mut hasher = Sha256::new();
            hasher.update(format!("{}{}{}{}{}", 
                timestamp, 
                previous_hash, 
                nonce, 
                transactions.len(),
                merkle_tree.root
            ));
            let hash = format!("{:x}", hasher.finalize());
            
            if hash.starts_with(&"0".repeat(self.difficulty as usize)) {
                return Block {
                    transactions: transactions.to_vec(),
                    timestamp,
                    previous_hash: previous_hash.to_string(),
                    hash,
                    difficulty: self.difficulty,
                    nonce,
                    merkle_root: merkle_tree.root,
                };
            }
            nonce += 1;
        }
    }

    fn create_block(&mut self) -> Option<Block> {
        if self.pending_transactions.is_empty() {
            return None;
        }

        let transactions = self.pending_transactions.drain(..).collect::<Vec<_>>();
        let previous_hash = self.blocks.last().map_or("0".repeat(64), |b| b.hash.clone());
        
        let block = self.mine_block(&transactions, &previous_hash);
        
        // Update balances
        for transaction in &block.transactions {
            if let Some(sender) = self.users.get_mut(&transaction.from) {
                sender.node.balance -= transaction.amount;
            }
            if let Some(receiver) = self.users.get_mut(&transaction.to) {
                receiver.node.balance += transaction.amount;
            }
        }

        self.blocks.push(block.clone());
        
        // Optimize regions after creating a block
        self.router.optimize_regions(&self.users);
        
        Some(block)
    }

    fn get_balance(&self, user_id: &str) -> f64 {
        self.users.get(user_id).map_or(0.0, |user| user.node.balance)
    }

    fn validate_chain(&self) -> bool {
        for i in 1..self.blocks.len() {
            let current = &self.blocks[i];
            let previous = &self.blocks[i - 1];
            
            if current.previous_hash != previous.hash {
                return false;
            }
            
            let mut hasher = Sha256::new();
            hasher.update(format!("{}{}{}{}", 
                current.timestamp, 
                current.previous_hash, 
                current.nonce,
                current.transactions.len()
            ));
            let hash = format!("{:x}", hasher.finalize());
            
            if hash != current.hash {
                return false;
            }
        }
        true
    }

    async fn start_network(&self) -> Result<(), Box<dyn Error>> {
        let mut swarm = SwarmBuilder::with_new_identity()
            .with_async_std()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                let peer_id = key.public().to_peer_id();
                
                // Configure gossipsub
                let gossipsub_config = GossipsubConfig::default();
                let gossipsub = gossipsub::Behaviour::new(
                    MessageAuthenticity::Signed(key.clone()),
                    gossipsub_config,
                )?;

                // Configure mdns
                let mdns = mdns::async_io::Behaviour::new(mdns::Config::default(), peer_id)?;

                Ok(MeshBehaviour { gossipsub, mdns })
            })?
            .build();

        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        Ok(())
    }

    async fn broadcast_transaction(&self, transaction: Transaction) -> Result<(), Box<dyn Error>> {
        let message = MeshMessage::Transaction(transaction.clone());
        let _encoded = serde_json::to_string(&message)?;
        
        let nodes_in_path = self.calculate_broadcast_path(&transaction);
        
        // Broadcast to nodes along the geodesic path
        for node_id in nodes_in_path {
            if let Some(user) = self.users.get(&node_id) {
                // Calculate the hyperbolic distance for routing priority
                let distance = self.hyperbolic_distance(
                    &user.node.position,
                    &self.get_transaction_position(&transaction)
                );
                
                // Add delay based on hyperbolic distance to create natural propagation
                let delay = (distance * 100.0) as u64;
                async_std::task::sleep(std::time::Duration::from_millis(delay)).await;
                
                // Actual broadcasting would happen here through the network layer
                println!("Broadcasting transaction to node {} at distance {}", node_id, distance);
            }
        }
        
        Ok(())
    }

    fn get_transaction_position(&self, transaction: &Transaction) -> Point {
        // Calculate the position in hyperbolic space based on sender and receiver
        let sender = self.users.get(&transaction.from).map(|u| &u.node.position);
        let receiver = self.users.get(&transaction.to).map(|u| &u.node.position);
        
        match (sender, receiver) {
            (Some(s), Some(r)) => Point {
                // Calculate midpoint in hyperbolic space
                x: (s.x + r.x) / (1.0 + s.x * r.x),
                y: (s.y + r.y) / (1.0 + s.y * r.y),
            },
            _ => Point { x: 0.0, y: 0.0 }
        }
    }

    fn calculate_broadcast_path(&self, transaction: &Transaction) -> Vec<String> {
        let mut path = Vec::new();
        let transaction_position = self.get_transaction_position(transaction);
        
        // Sort nodes by their hyperbolic distance to the transaction
        let mut nodes: Vec<_> = self.users.iter().collect();
        nodes.sort_by(|a, b| {
            let dist_a = self.hyperbolic_distance(&a.1.node.position, &transaction_position);
            let dist_b = self.hyperbolic_distance(&b.1.node.position, &transaction_position);
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        
        // Select nodes that form an efficient broadcast path
        for (id, _) in nodes.iter().take(((self.users.len() as f64).sqrt() as usize).max(1)) {
            path.push(id.to_string());
        }
        
        path
    }

    async fn broadcast_block(&self, block: Block) -> Result<(), Box<dyn Error>> {
        let message = MeshMessage::Block(block.clone());
        let _encoded = serde_json::to_string(&message)?;
        
        let block_position = self.calculate_block_position(&block);
        let broadcast_nodes = self.get_broadcast_nodes(&block_position);
        
        // Broadcast in waves based on hyperbolic distance
        for node_batch in broadcast_nodes.chunks(5) {
            for node_id in node_batch {
                if let Some(user) = self.users.get(node_id) {
                    let distance = self.hyperbolic_distance(&user.node.position, &block_position);
                    let delay = (distance * 50.0) as u64;
                    async_std::task::sleep(std::time::Duration::from_millis(delay)).await;
                    
                    println!("Broadcasting block to node {} at distance {}", node_id, distance);
                }
            }
        }
        
        Ok(())
    }

    

    fn get_broadcast_nodes(&self, position: &Point) -> Vec<String> {
        let mut nodes: Vec<_> = self.users.keys().cloned().collect();
        nodes.sort_by(|a, b| {
            let pos_a = &self.users.get(a).unwrap().node.position;
            let pos_b = &self.users.get(b).unwrap().node.position;
            let dist_a = self.hyperbolic_distance(pos_a, position);
            let dist_b = self.hyperbolic_distance(pos_b, position);
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        nodes
    }

    pub async fn register_user(&mut self, username: String) -> Result<(), Box<dyn Error>> {
        let id_keys = identity::Keypair::generate_ed25519();
        
        let user = User {
            id: username.clone(),
            keypair: id_keys,
            node: Node {
                position: Point { 
                    x: rand::random::<f64>() * 2.0 - 1.0,
                    y: rand::random::<f64>() * 2.0 - 1.0
                },
                connections: Vec::new(),
                transactions: Vec::new(),
                balance: 0.0,
                is_validator: false,  // Default to false
            },
            serialized_keypair: None,
            quantum_crypto: Arc::new(QuantumCrypto::new()?),
        };

        self.users.insert(username, user);
        Ok(())
    }

    pub async fn start_user_node(&mut self, username: &str) -> Result<(), Box<dyn Error>> {
        // Clone the user data before using it to avoid borrow conflicts
        let user = if let Some(user) = self.users.get(username) {
            user.clone()
        } else {
            return Err("User not found".into());
        };
        
        // Now use the cloned user data
        self.start_network_for_user(&user).await?;
        Ok(())
    }

    pub fn create_transaction(&mut self, 
        from_user: &str, 
        to_user: &str, 
        amount: f64
    ) -> Result<(), Box<dyn Error>> {
        let sender = self.users.get(from_user)
            .ok_or("Sender not found")?;
        
        let transaction = Transaction {
            from: from_user.to_string(),
            to: to_user.to_string(),
            amount,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            signature: self.sign_transaction(sender, amount)?,
        };

        // Use block_on for synchronous context
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(async {
                if self.validate_transaction(&transaction).await? {
                    self.pending_transactions.push_back(transaction);
                    Ok(())
                } else {
                    Err("Invalid transaction".into())
                }
            })
    }

    fn sign_transaction(&self, user: &User, amount: f64) -> Result<String, Box<dyn Error>> {
        // Implement proper transaction signing using the user's keypair
        // This is a simplified version
        Ok(format!("signed_{}_{}", user.id, amount))
    }

    async fn start_network_for_user(&mut self, user: &User) -> Result<(), Box<dyn Error>> {
        let peer_id = PeerId::from(user.keypair.public());
        println!("Starting node for user {} with peer id: {}", user.id, peer_id);

        let mut swarm = SwarmBuilder::with_new_identity()
            .with_async_std()
            .with_tcp(
                tcp::Config::default(),
                noise::Config::new,
                yamux::Config::default,
            )?
            .with_behaviour(|key| {
                let peer_id = key.public().to_peer_id();
                
                // Configure gossipsub
                let gossipsub_config = gossipsub::ConfigBuilder::default()
                    .heartbeat_interval(Duration::from_secs(1))
                    .validation_mode(gossipsub::ValidationMode::Strict)
                    .build()
                    .expect("Valid config");

                let mut gossipsub = gossipsub::Behaviour::new(
                    MessageAuthenticity::Signed(user.keypair.clone()),
                    gossipsub_config,
                )?;

                // Create and subscribe to topics
                let transactions_topic = gossipsub::IdentTopic::new("transactions");
                let blocks_topic = gossipsub::IdentTopic::new("blocks");
                let chain_topic = gossipsub::IdentTopic::new("chain");

                gossipsub.subscribe(&transactions_topic)?;
                gossipsub.subscribe(&blocks_topic)?;
                gossipsub.subscribe(&chain_topic)?;

                // Configure mdns
                let mdns = mdns::async_io::Behaviour::new(mdns::Config::default(), peer_id)?;

                Ok(MeshBehaviour { gossipsub, mdns })
            })?
            .build();

        swarm.listen_on("/ip4/0.0.0.0/tcp/0".parse()?)?;

        // Add timer for periodic region optimization
        let mut optimization_interval = tokio::time::interval(Duration::from_secs(300)); // every 5 minutes

        // Handle network events
        loop {
            tokio::select! {
                _ = optimization_interval.tick() => {
                    // Periodic region optimization
                    self.router.optimize_regions(&self.users);
                }
                event = swarm.next() => match event {
                    Some(SwarmEvent::Behaviour(MeshBehaviourEvent::Gossipsub(
                        gossipsub::Event::Message {
                            propagation_source: peer_id,
                            message_id: _id,  // Prefix with underscore
                            message,
                        }
                    ))) => {
                        self.handle_message(message, peer_id).await?;
                    }
                    Some(SwarmEvent::Behaviour(MeshBehaviourEvent::Mdns(
                        mdns::Event::Discovered(list)
                    ))) => {
                        for (peer_id, _multiaddr) in list {
                            println!("mDNS discovered peer: {}", peer_id);
                        }
                    }
                    Some(SwarmEvent::NewListenAddr { address, .. }) => {
                        println!("Listening on {}", address);
                    }
                    _ => {}
                }
            }
        }
    }

    async fn handle_message(
        &mut self,
        message: gossipsub::Message,
        source: PeerId,
    ) -> Result<(), Box<dyn Error>> {
        self.metrics.messages_received += 1;
        
        let mesh_message: MeshMessage = serde_json::from_slice(&message.data)?;
        let message_id = format!("{:x}", Sha256::digest(&message.data));
        
        // Send acknowledgment
        self.send_ack(&message_id, &source).await?;
        
        match mesh_message {
            MeshMessage::Transaction(ref transaction) => {
                if self.validate_propagating_transaction(transaction).await? {
                    self.forward_transaction(transaction).await?;
                } else {
                    self.handle_failed_validation(&mesh_message).await?;
                }
            }
            MeshMessage::Block(ref block) => {
                if self.validate_propagating_block(block).await? {
                    self.forward_block(block).await?;
                } else {
                    self.handle_failed_validation(&mesh_message).await?;
                }
            }
            MeshMessage::SyncStatus { height, peer_id } => {
                self.sync_state.insert(peer_id, height);
                self.synchronize_chain(&source.to_string()).await?;
            }
            MeshMessage::MessageAck { message_id } => {
                self.pending_messages.remove(&message_id);
            }
            MeshMessage::RequestChain { from_block } => {
                // Handle chain request
                if let Some(blocks) = self.get_blocks_from(from_block) {
                    let response = MeshMessage::ChainResponse { blocks };
                    self.broadcast_message(&response).await?;
                }
            }
            MeshMessage::ChainResponse { blocks } => {
                // Handle chain response
                self.process_chain_response(blocks).await?;
            }
            MeshMessage::RetransmissionRequest { message_id } => {
                // Clone the message before borrowing self as mutable
                let message_to_send = if let Some((message, _)) = self.pending_messages.get(&message_id) {
                    message.clone()
                } else {
                    return Ok(());
                };
                self.broadcast_message(&message_to_send).await?;
            }
            MeshMessage::EncryptedTransaction { ciphertext, sender_id } => {
                self.handle_encrypted_message(ciphertext, sender_id).await?;
            }
            MeshMessage::QuantumKeyExchange { public_key: _public_key, sender_id } => {
                if let Some(_receiver) = self.users.get(&self.local_peer_id.unwrap().to_string()) {
                    // Process the quantum key exchange
                    println!("Received quantum key exchange from {}", sender_id);
                }
            }
        }
        
        Ok(())
    }

    async fn validate_propagating_transaction(&self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        // 1. Basic validation
        if !self.validate_transaction_basics(transaction) {
            return Ok(false);
        }

        // 2. Cryptographic validation
        if !self.validate_transaction_signature(transaction)? {
            return Ok(false);
        }

        // 3. Position validation in hyperbolic space
        let position = self.get_transaction_position(transaction);
        if !self.validate_transaction_position(transaction, &position) {
            return Ok(false);
        }

        // 4. Double-spend validation
        if self.is_double_spend(transaction).await? {
            return Ok(false);
        }

        // 5. Balance validation
        if !self.validate_sender_balance(transaction) {
            return Ok(false);
        }

        // 6. Zero-knowledge proof validation
        if !self.validate_transaction_proof(transaction).await? {
            return Ok(false);
        }

        Ok(true)
    }

    fn validate_transaction_basics(&self, transaction: &Transaction) -> bool {
        // Check basic transaction properties
        if transaction.amount <= 0.0 {
            return false;
        }
        if transaction.from == transaction.to {
            return false;
        }
        if !self.users.contains_key(&transaction.from) || !self.users.contains_key(&transaction.to) {
            return false;
        }
        true
    }

    fn validate_transaction_signature(&self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        // Verify the transaction signature using the sender's public key
        if let Some(_sender) = self.users.get(&transaction.from) {
            // In a real implementation, this would use proper cryptographic verification
            Ok(transaction.signature.starts_with("signed_"))
        } else {
            Ok(false)
        }
    }

    async fn is_double_spend(&self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        // Check if transaction is already in pending transactions
        if self.pending_transactions.iter().any(|t| t.signature == transaction.signature) {
            return Ok(true);
        }

        // Check if transaction is already in a block
        for block in &self.blocks {
            if block.transactions.iter().any(|t| t.signature == transaction.signature) {
                return Ok(true);
            }
        }

        Ok(false)
    }

    fn validate_sender_balance(&self, transaction: &Transaction) -> bool {
        if let Some(sender) = self.users.get(&transaction.from) {
            sender.node.balance >= transaction.amount
        } else {
            false
        }
    }
    

    async fn validate_propagating_block(&self, block: &Block) -> Result<bool, Box<dyn Error>> {
        // 1. Basic block validation
        if !self.validate_block_basics(block) {
            return Ok(false);
        }

        // 2. Position validation in hyperbolic space
        let position = self.calculate_block_position(block);
        if !self.validate_block_position(block, &position) {
            return Ok(false);
        }

        // 3. Merkle root validation
        if !self.validate_block_merkle_root(block) {
            return Ok(false);
        }

        // 4. Transaction validation within block
        if !self.validate_block_transactions(block).await? {
            return Ok(false);
        }

        // 5. Proof of work validation
        if !self.validate_block_pow(block) {
            return Ok(false);
        }

        Ok(true)
    }

    fn validate_block_basics(&self, block: &Block) -> bool {
        // Check basic block properties
        if block.transactions.is_empty() {
            return false;
        }
        if block.timestamp == 0 {
            return false;
        }
        if block.previous_hash.len() != 64 {
            return false;
        }
        true
    }

    fn validate_block_merkle_root(&self, block: &Block) -> bool {
        let merkle_tree = MerkleTree::new(&block.transactions);
        merkle_tree.root == block.merkle_root
    }

    async fn validate_block_transactions(&self, block: &Block) -> Result<bool, Box<dyn Error>> {
        // Validate each transaction in the block
        for transaction in &block.transactions {
            if !self.validate_propagating_transaction(transaction).await? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn validate_block_pow(&self, block: &Block) -> bool {
        // Verify the proof of work
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}{}{}", 
            block.timestamp, 
            block.previous_hash, 
            block.nonce,
            block.transactions.len()
        ));
        let hash = format!("{:x}", hasher.finalize());
        
        hash.starts_with(&"0".repeat(block.difficulty as usize))
    }

    async fn forward_transaction(&self, transaction: &Transaction) -> Result<(), Box<dyn Error>> {
        // Calculate optimal forwarding path based on hyperbolic geometry
        let position = self.get_transaction_position(transaction);
        let forward_nodes = self.calculate_forward_nodes(&position);
        
        for node_id in forward_nodes {
            // Add propagation delay based on hyperbolic distance
            let delay = self.calculate_propagation_delay(&node_id, &position);
            async_std::task::sleep(Duration::from_millis(delay)).await;
            
            // Forward the transaction
            println!("Forwarding transaction to node {}", node_id);
        }
        
        Ok(())
    }

    async fn forward_block(&self, block: &Block) -> Result<(), Box<dyn Error>> {
        // Calculate optimal forwarding path based on hyperbolic geometry
        let position = self.calculate_block_position(block);
        let forward_nodes = self.calculate_forward_nodes(&position);
        
        for node_id in forward_nodes {
            // Add propagation delay based on hyperbolic distance
            let delay = self.calculate_propagation_delay(&node_id, &position);
            async_std::task::sleep(Duration::from_millis(delay)).await;
            
            // Forward the block
            println!("Forwarding block to node {}", node_id);
        }
        
        Ok(())
    }

    fn calculate_forward_nodes(&self, position: &Point) -> Vec<String> {
        // Select nodes for forwarding based on hyperbolic distance and network topology
        let mut nodes: Vec<_> = self.users.keys().cloned().collect();
        nodes.sort_by(|a, b| {
            let dist_a = self.hyperbolic_distance(&self.users.get(a).unwrap().node.position, position);
            let dist_b = self.hyperbolic_distance(&self.users.get(b).unwrap().node.position, position);
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        
        // Take sqrt(n) closest nodes for forwarding
        nodes.into_iter()
            .take(((self.users.len() as f64).sqrt() as usize).max(1))
            .collect()
    }

    fn calculate_propagation_delay(&self, node_id: &str, position: &Point) -> u64 {
        if let Some(user) = self.users.get(node_id) {
            let distance = self.hyperbolic_distance(&user.node.position, position);
            (distance * 100.0) as u64
        } else {
            0
        }
    }

    fn validate_transaction_position(&self, transaction: &Transaction, position: &Point) -> bool {
        // Validate that the transaction position makes sense in hyperbolic space
        let sender_pos = self.users.get(&transaction.from)
            .map(|u| &u.node.position)
            .unwrap_or(&Point { x: 0.0, y: 0.0 });
        
        let receiver_pos = self.users.get(&transaction.to)
            .map(|u| &u.node.position)
            .unwrap_or(&Point { x: 0.0, y: 0.0 });

        // Check if position lies on the geodesic path between sender and receiver
        let max_distance = self.hyperbolic_distance(sender_pos, receiver_pos);
        let actual_distance = self.hyperbolic_distance(sender_pos, position) +
                            self.hyperbolic_distance(position, receiver_pos);
        
        // Allow for some numerical error
        (actual_distance - max_distance).abs() < 0.001
    }

    fn validate_block_position(&self, block: &Block, position: &Point) -> bool {
        // Validate that the block position makes sense given its transactions
        let mut total_distance = 0.0;
        let mut count = 0;

        for tx in &block.transactions {
            let tx_pos = self.get_transaction_position(tx);
            total_distance += self.hyperbolic_distance(&tx_pos, position);
            count += 1;
        }

        if count == 0 {
            return true;
        }

        // Check if the block position is reasonably central to its transactions
        let avg_distance = total_distance / count as f64;
        avg_distance < 1.0 // Adjust threshold as needed
    }

    async fn handle_message_reliability(&mut self, message_id: String, message: MeshMessage) -> Result<(), Box<dyn Error>> {
        const MAX_RETRIES: u32 = 3;
        const RETRY_DELAY_MS: u64 = 1000;

        self.pending_messages.insert(message_id.clone(), (message.clone(), 0));
        
        // Wait for acknowledgment
        for retry in 0..MAX_RETRIES {
            if !self.pending_messages.contains_key(&message_id) {
                return Ok(());
            }
            
            // Retransmit if no ack received
            if retry > 0 {
                println!("Retransmitting message {}, attempt {}", message_id, retry);
                self.metrics.retransmissions += 1;
                // Rebroadcast message
                self.broadcast_message(&message).await?;
            }
            
            async_std::task::sleep(Duration::from_millis(RETRY_DELAY_MS)).await;
        }
        
        // Message failed after max retries
        self.handle_failed_transmission(&message_id).await?;
        Ok(())
    }

    async fn synchronize_chain(&mut self, peer_id: &str) -> Result<(), Box<dyn Error>> {
        self.metrics.sync_requests += 1;
        
        // Get peer's chain height
        if let Some(&peer_height) = self.sync_state.get(peer_id) {
            let our_height = self.blocks.len();
            
            if peer_height > our_height {
                // Request missing blocks
                let message = MeshMessage::RequestChain {
                    from_block: our_height
                };
                self.broadcast_message(&message).await?;
                
                // Log sync attempt
                println!("Initiating chain sync with peer {}: local height {}, peer height {}", 
                    peer_id, our_height, peer_height);
            }
        }
        
        Ok(())
    }

    async fn handle_failed_validation(&mut self, message: &MeshMessage) -> Result<(), Box<dyn Error>> {
        self.metrics.failed_validations += 1;
        
        match message {
            MeshMessage::Transaction(tx) => {
                println!("Transaction validation failed: {:?}", tx);
                // Request recent state from peers
                self.request_state_update().await?;
            }
            MeshMessage::Block(block) => {
                println!("Block validation failed: {:?}", block);
                // Trigger chain synchronization
                self.synchronize_chain_with_peers().await?;
            }
            _ => {}
        }
        
        Ok(())
    }

    fn update_metrics(&mut self, event: MetricEvent) {
        match event {
            MetricEvent::MessageSent => self.metrics.messages_sent += 1,
            MetricEvent::MessageReceived => self.metrics.messages_received += 1,
            MetricEvent::ValidationFailed => self.metrics.failed_validations += 1,
            MetricEvent::SyncRequested => self.metrics.sync_requests += 1,
        }
    }

    pub fn get_metrics(&self) -> &NetworkMetrics {
        &self.metrics
    }

    async fn send_ack(&mut self, message_id: &str, source: &PeerId) -> Result<(), Box<dyn Error>> {
        let ack_message = MeshMessage::MessageAck {
            message_id: message_id.to_string(),
        };
        
        let encoded = serde_json::to_string(&ack_message)?;
        
        if let Some(swarm) = &mut self.swarm {
            swarm.behaviour_mut().gossipsub.publish(
                gossipsub::IdentTopic::new("acks"),
                encoded.as_bytes(),
            )?;
            
            println!("Sent ACK for message {} to peer {}", message_id, source);
        }
        
        Ok(())
    }

    async fn broadcast_message(&mut self, message: &MeshMessage) -> Result<(), Box<dyn Error>> {
        let encoded = serde_json::to_string(message)?;
        
        let source_position = match message {
            MeshMessage::Transaction(tx) => self.get_transaction_position(tx),
            MeshMessage::Block(block) => self.calculate_block_position(block),
            _ => Point { x: 0.0, y: 0.0 },
        };
        
        // Extract necessary data from self before calling update_routing_table
        let users_clone = self.users.clone();
        self.router.update_routing_table(&users_clone);
        
        // Get optimal paths for broadcasting
        let mut processed_nodes = HashSet::new();
        for (node_id, user) in &self.users {
            if processed_nodes.contains(node_id) {
                continue;
            }
            
            let path = self.router.calculate_geodesic_path(
                &source_position,
                &user.node.position
            );
            
            // Follow the geodesic path for message propagation
            for point in path {
                // Find closest node to this point
                if let Some((closest_node, _)) = self.find_closest_node(&point) {
                    if !processed_nodes.contains(&closest_node) {
                        processed_nodes.insert(closest_node.clone());
                        
                        // Calculate delay based on hyperbolic distance
                        let delay = self.calculate_propagation_delay(&closest_node, &point);
                        async_std::task::sleep(Duration::from_millis(delay)).await;
                        
                        // Broadcast through the swarm
                        if let Some(swarm) = &mut self.swarm {
                            swarm.behaviour_mut().gossipsub.publish(
                                gossipsub::IdentTopic::new("messages"),
                                encoded.as_bytes(),
                            )?;
                        }
                    }
                }
            }
        }
        
        Ok(())
    }

    fn find_closest_node(&self, point: &Point) -> Option<(String, f64)> {
        self.router.routing_table.keys()
            .map(|id| (id.clone(), self.hyperbolic_distance(point, &Point { x: 0.0, y: 0.0 }))) // You'll need to get actual node positions
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
    }

    async fn handle_failed_transmission(&mut self, message_id: &str) -> Result<(), Box<dyn Error>> {
        println!("Transmission failed for message: {}", message_id);
        
        // Get the failed message and its retry count
        if let Some((message, retry_count)) = self.pending_messages.get(message_id) {
            println!("Message failed after {} retries", retry_count);
            
            // Clone the message before using it
            let message_clone = message.clone();
            
            match message_clone {
                MeshMessage::Transaction(tx) => {
                    // For failed transactions, try alternative paths
                    let position = self.get_transaction_position(&tx);
                    let alternative_nodes = self.calculate_alternative_paths(&position);
                    
                    // Attempt to send through alternative paths
                    for node_id in alternative_nodes {
                        if let Some(_user) = self.users.get(node_id.as_str()) {
                            println!("Attempting alternative path through node {}", node_id);
                            // Implementation would depend on your network layer
                        }
                    }
                },
                MeshMessage::Block(_block) => {
                    // For failed blocks, trigger state synchronization
                    println!("Block transmission failed, initiating state sync");
                    self.request_state_update().await?;
                },
                _ => {
                    println!("Generic message transmission failed");
                }
            }
        }
        
        Ok(())
    }

    async fn request_state_update(&mut self) -> Result<(), Box<dyn Error>> {
        // Get our current chain height
        let our_height = self.blocks.len();
        let local_peer_id = self.local_peer_id.expect("Peer ID not set").to_string();
        
        // Create a sync status message
        let sync_message = MeshMessage::SyncStatus {
            height: our_height,
            peer_id: local_peer_id,
        };
        
        // Get peers with higher block height
        let peers_to_sync: Vec<_> = self.sync_state.iter()
            .filter(|(_, &height)| height > our_height)
            .map(|(peer_id, _)| peer_id.clone())
            .collect();
        
        // Broadcast our status
        self.broadcast_message(&sync_message).await?;
        
        // Request chain from each peer
        for peer_id in peers_to_sync {
            let request = MeshMessage::RequestChain {
                from_block: our_height,
            };
            
            if let Some(peer_user) = self.users.get(peer_id.as_str()) {
                let distance = self.hyperbolic_distance(
                    &peer_user.node.position,
                    &Point { x: 0.0, y: 0.0 }
                );
                
                let delay = (distance * 100.0) as u64;
                async_std::task::sleep(Duration::from_millis(delay)).await;
                
                if let Some(swarm) = &mut self.swarm {
                    let encoded = serde_json::to_string(&request)?;
                    swarm.behaviour_mut().gossipsub.publish(
                        gossipsub::IdentTopic::new("chain_sync"),
                        encoded.as_bytes(),
                    )?;
                }
            }
        }
        
        Ok(())
    }

    async fn synchronize_chain_with_peers(&mut self) -> Result<(), Box<dyn Error>> {
        let mut peer_heights: Vec<_> = self.sync_state.iter().collect();
        peer_heights.sort_by_key(|&(_, height)| *height);
        
        let median_height = if let Some(&(_, height)) = peer_heights.get(peer_heights.len() / 2) {
            *height
        } else {
            return Ok(());
        };
        
        let our_height = self.blocks.len();
        
        if our_height < median_height {
            let peers_to_sync = peer_heights.iter()
                .rev()
                .take(3)
                .map(|(peer_id, _)| peer_id.clone())
                .collect::<Vec<_>>();
            
            for peer_id in peers_to_sync {
                let request = MeshMessage::RequestChain {
                    from_block: our_height,
                };
                
                if let Some(peer_user) = self.users.get(peer_id.as_str()) {
                    let distance = self.hyperbolic_distance(
                        &peer_user.node.position,
                        &Point { x: 0.0, y: 0.0 }
                    );
                    
                    let delay = (distance * 50.0) as u64;
                    async_std::task::sleep(Duration::from_millis(delay)).await;
                    
                    if let Some(swarm) = &mut self.swarm {
                        let encoded = serde_json::to_string(&request)?;
                        swarm.behaviour_mut().gossipsub.publish(
                            gossipsub::IdentTopic::new("chain_sync"),
                            encoded.as_bytes(),
                        )?;
                    }
                }
            }
        }
        
        Ok(())
    }

    // Helper method for calculating alternative paths
    fn calculate_alternative_paths(&self, position: &Point) -> Vec<String> {
        let mut nodes = self.get_broadcast_nodes(position);
        
        // Filter out nodes that were likely used in the original transmission
        nodes.retain(|node_id| {
            if let Some(user) = self.users.get(node_id) {
                // Calculate distance to determine if this was likely in original path
                let distance = self.hyperbolic_distance(&user.node.position, position);
                // Use nodes that are slightly further away as alternatives
                distance > 1.0 && distance < 2.0
            } else {
                false
            }
        });
        
        nodes
    }

    fn get_blocks_from(&self, from_block: usize) -> Option<Vec<Block>> {
        if from_block <= self.blocks.len() {
            Some(self.blocks[from_block..].to_vec())
        } else {
            None
        }
    }

    async fn process_chain_response(&mut self, blocks: Vec<Block>) -> Result<(), Box<dyn Error>> {
        // Validate the received blocks
        for block in &blocks {
            if !self.validate_propagating_block(block).await? {
                return Err("Invalid block in chain response".into());
            }
        }

        // Update our chain if the received blocks are valid
        let current_height = self.blocks.len();
        if blocks.first().map_or(false, |b| {
            b.previous_hash == self.blocks.last().map_or("0".repeat(64), |lb| lb.hash.clone())
        }) {
            self.blocks.extend(blocks);
            println!("Chain updated from height {} to {}", current_height, self.blocks.len());
        }

        Ok(())
    }

    fn communicate_across_shards(&self, transaction: &Transaction) {
        let from_shard = self.router.find_shard_for_node(&transaction.from);
        let to_shard = self.router.find_shard_for_node(&transaction.to);

        if from_shard != to_shard {
            if let (Some(from_shard), Some(to_shard)) = (
                self.router.shards.get(from_shard),
                self.router.shards.get(to_shard)
            ) {
                // Calculate geodesic path between shards
                let path = self.router.calculate_geodesic_path(
                    &from_shard.center,
                    &to_shard.center,
                );

                // Propagate transaction along the path
                for point in path {
                    if let Some((closest_node, _)) = self.router.find_closest_node(&point) {
                        println!("Propagating transaction across shard boundary via node {}", closest_node);
                    }
                }
            }
        }
    }

    fn compress_state(&self) -> Result<CompressedState, Box<dyn Error>> {
        // Generate a unique key for the state based on current state points
        let state_points = self.get_state_points();
        let mut hasher = Sha256::new();
        for point in &state_points {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        let state_key = format!("state_{:x}", hasher.finalize());

        // Check if the compressed state is already cached
        if let Some(cached_state) = self.get_compressed_state(&state_key) {
            return Ok(cached_state);
        }

        // Get the current state points in hyperbolic space
        if state_points.is_empty() {
            return Err("No state points available for compression".into());
        }
        
        // Calculate optimal geodesic path for compression
        let geodesic_path = self.calculate_compression_path(&state_points);
        
        // Compress state along geodesic path
        let (compressed_data, ratio) = self.compress_along_geodesic(&geodesic_path)?;
        
        // Generate geometric proof for compressed state
        let proof = self.generate_geometric_proof(&geodesic_path, &compressed_data)?;
        
        let compressed_state = CompressedState {
            geodesic_path,
            compressed_data,
            compression_ratio: ratio,
            proof,
        };

        // Cache the newly compressed state
        self.cache_compressed_state(state_key, compressed_state.clone());

        Ok(compressed_state)
    }

    fn get_state_points(&self) -> Vec<Point> {
        let mut points = Vec::new();
        
        // Collect points from transactions and blocks
        for block in &self.blocks {
            for tx in &block.transactions {
                points.push(self.get_transaction_position(tx));
            }
        }
        
        // Add points from pending transactions
        for tx in &self.pending_transactions {
            points.push(self.get_transaction_position(tx));
        }
        
        points
    }

    fn calculate_compression_path(&self, points: &[Point]) -> Vec<Point> {
        let mut path = Vec::new();
        
        // Skip if no points
        if points.is_empty() {
            return path;
        }
        
        // Start with the center of mass
        let mut current = self.calculate_center_of_mass(points);
        path.push(current);
        
        // Find optimal path through points using hyperbolic distance
        let mut remaining: Vec<_> = points.to_vec();
        while !remaining.is_empty() {
            // Find closest point in hyperbolic space
            if let Some((next_idx, _)) = remaining.iter().enumerate()
                .min_by(|(_, a), (_, b)| {
                    let dist_a = self.hyperbolic_distance(&current, a);
                    let dist_b = self.hyperbolic_distance(&current, b);
                    dist_a.partial_cmp(&dist_b).unwrap()
                }) 
            {
                current = remaining.remove(next_idx);
                path.push(current);
            }
        }
        
        path
    }

    fn calculate_center_of_mass(&self, points: &[Point]) -> Point {
        let mut x_sum = 0.0;
        let mut y_sum = 0.0;
        let count = points.len() as f64;
        
        for point in points {
            x_sum += point.x;
            y_sum += point.y;
        }
        
        Point {
            x: x_sum / count,
            y: y_sum / count,
        }
    }

    fn compress_along_geodesic(&self, path: &[Point]) -> Result<(Vec<u8>, f64), Box<dyn Error>> {
        let mut compressed = Vec::new();
        let original_size = path.len() * std::mem::size_of::<Point>();
        
        // Group points by their position along the geodesic
        let mut segments = Vec::new();
        let mut current_segment = Vec::new();
        
        for window in path.windows(2) {
            let distance = self.hyperbolic_distance(&window[0], &window[1]);
            if distance < 0.1 { // Adjustable threshold
                current_segment.push(window[1]);
            } else {
                if !current_segment.is_empty() {
                    segments.push(current_segment);
                }
                current_segment = vec![window[1]];
            }
        }
        
        // Add last segment
        if !current_segment.is_empty() {
            segments.push(current_segment);
        }
        
        // Compress each segment
        for segment in segments {
            let avg_point = self.calculate_center_of_mass(&segment);
            compressed.extend_from_slice(&avg_point.x.to_le_bytes());
            compressed.extend_from_slice(&avg_point.y.to_le_bytes());
        }
        
        let compression_ratio = original_size as f64 / compressed.len() as f64;
        Ok((compressed, compression_ratio))
    }

    fn generate_geometric_proof(&self, path: &[Point], compressed: &[u8]) -> Result<GeometricProof, Box<dyn Error>> {
        // Generate tessellation points for verification
        let tessellation = self.generate_tessellation(path);
        
        // Calculate verification path through tessellation
        let verification_path = self.calculate_verification_path(&tessellation, compressed);
        
        // Generate proof hash
        let mut hasher = Sha256::new();
        for point in &verification_path {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        hasher.update(compressed);
        let proof_hash = format!("{:x}", hasher.finalize());
        
        Ok(GeometricProof {
            tessellation_points: tessellation,
            verification_path,
            position_data: Vec::new(),
            hash: proof_hash,
            signature: Vec::new(),
            timestamp: 0,
        })
    }

    fn generate_tessellation(&self, path: &[Point]) -> Vec<Point> {
        self.generate_optimized_tessellation(path)
    }

    fn generate_optimized_tessellation(&self, path: &[Point]) -> Vec<Point> {
        let mut tessellation = Vec::new();
        let mut covered_areas = HashSet::new();
        
        // Calculate optimal tessellation density based on path length
        let density = self.calculate_tessellation_density(path);
        
        for &center in path {
            // Skip if area is already well-covered
            let area_key = self.get_area_key(&center);
            if covered_areas.contains(&area_key) {
                continue;
            }
            
            // Generate adaptive tessellation around this point
            let local_points = self.generate_adaptive_tessellation(&center, density);
            tessellation.extend(local_points);
            
            // Mark area as covered
            covered_areas.insert(area_key);
        }
        
        // Optimize tessellation by removing redundant points
        self.optimize_tessellation_points(&mut tessellation);
        
        tessellation
    }
    
    fn calculate_tessellation_density(&self, path: &[Point]) -> f64 {
        // Calculate path length in hyperbolic space
        let mut total_length = 0.0;
        for window in path.windows(2) {
            total_length += self.hyperbolic_distance(&window[0], &window[1]);
        }
        
        // Adjust density based on path length and curvature
        let base_density = 0.1;
        let curvature_factor = (-total_length).exp();
        
        base_density * (1.0 + curvature_factor)
    }
    
    fn generate_adaptive_tessellation(&self, center: &Point, density: f64) -> Vec<Point> {
        let mut points = Vec::new();
        let layers = (1.0 / density).ceil() as i32;
        
        // Generate concentric layers of tessellation
        for layer in 1..=layers {
            let radius = density * layer as f64;
            let points_in_layer = (2.0 * std::f64::consts::PI * layer as f64) as i32;
            
            for i in 0..points_in_layer {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (points_in_layer as f64);
                
                // Calculate point position in hyperbolic space
                let (x, y) = self.hyperbolic_polar_to_cartesian(center, radius, angle);
                
                // Ensure point is within Poincaré disk
                if x * x + y * y < 1.0 {
                    points.push(Point { x, y });
                }
            }
        }
        
        points
    }
    
    fn hyperbolic_polar_to_cartesian(&self, center: &Point, radius: f64, angle: f64) -> (f64, f64) {
        // Convert hyperbolic polar coordinates to Poincaré disk coordinates
        let tanh_r = radius.tanh();
        let x = tanh_r * angle.cos();
        let y = tanh_r * angle.sin();
        
        // Apply Möbius transformation to center at given point
        let denom = 1.0 + 2.0 * (center.x * x + center.y * y) + (x * x + y * y);
        let x_centered = (x + center.x) / denom;
        let y_centered = (y + center.y) / denom;
        
        (x_centered, y_centered)
    }
    
    fn optimize_tessellation_points(&self, points: &mut Vec<Point>) {
        let min_distance = 0.05; // Minimum allowed distance between points
        
        // Sort points by distance from origin for deterministic results
        points.sort_by(|a, b| {
            let dist_a = a.x * a.x + a.y * a.y;
            let dist_b = b.x * b.x + b.y * b.y;
            dist_a.partial_cmp(&dist_b).unwrap()
        });
        
        // Remove points that are too close to others
        let mut i = 0;
        while i < points.len() {
            let mut j = i + 1;
            while j < points.len() {
                if self.hyperbolic_distance(&points[i], &points[j]) < min_distance {
                    points.remove(j);
                } else {
                    j += 1;
                }
            }
            i += 1;
        }
    }
    
    fn get_area_key(&self, point: &Point) -> (i32, i32) {
        // Discretize space into grid cells for coverage tracking
        let cell_size = 0.1;
        (
            (point.x / cell_size).floor() as i32,
            (point.y / cell_size).floor() as i32
        )
    }

    fn calculate_verification_path(&self, tessellation: &[Point], compressed_data: &[u8]) -> Vec<Point> {
        let mut path = Vec::new();
        let mut current = Point { x: 0.0, y: 0.0 }; // Start at origin
        path.push(current);
        
        // Extract target points from compressed data
        let mut target_points = Vec::new();
        for chunk in compressed_data.chunks(16) {
            if chunk.len() >= 16 {
                let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
                let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
                if x.is_finite() && y.is_finite() {
                    target_points.push(Point { x, y });
                }
            }
        }
        
        // For each target point, create a continuous path through tessellation points
        for target in target_points {
            // Sort tessellation points by distance from current position
            let mut available_points: Vec<_> = tessellation.iter()
                .filter(|&p| {
                    let dist_to_target = self.hyperbolic_distance(p, &target);
                    let dist_to_current = self.hyperbolic_distance(&current, &target);
                    dist_to_target.is_finite() && dist_to_current.is_finite() && dist_to_target < dist_to_current
                })
                .collect();
            
            available_points.sort_by(|&a, &b| {
                let dist_a = self.hyperbolic_distance(&current, a);
                let dist_b = self.hyperbolic_distance(&current, b);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            });
            
            // Add intermediate points to ensure continuity
            for point in available_points {
                let distance = self.hyperbolic_distance(&current, point);
                if distance.is_finite() && distance < 0.3 {
                    path.push(*point);
                    current = *point;
                }
            }
            
            // Add interpolated points if necessary
            if self.hyperbolic_distance(&current, &target) > 0.3 {
                let steps = (self.hyperbolic_distance(&current, &target) / 0.2).ceil() as i32;
                for i in 1..steps {
                    let t = i as f64 / steps as f64;
                    let interpolated = self.interpolate_points(&current, &target, t);
                    if interpolated.x.is_finite() && interpolated.y.is_finite() {
                        path.push(interpolated);
                    }
                }
            }
            
            // Add the target point
            path.push(target);
            current = target;
        }
        
        path
    }
    
    // Helper method to interpolate between two points in hyperbolic space
    fn interpolate_points(&self, start: &Point, end: &Point, t: f64) -> Point {
        let dx = end.x - start.x;
        let dy = end.y - start.y;
        
        // Linear interpolation with normalization to ensure points stay in disk
        let x = start.x + dx * t;
        let y = start.y + dy * t;
        
        let norm = (x * x + y * y).sqrt();
        if norm > 1.0 {
            let scale = 0.99 / norm;
            Point {
                x: x * scale,
                y: y * scale,
            }
        } else {
            Point { x, y }
        }
    }

    // Add this new method for state reconstruction
    fn reconstruct_state(&self, compressed: &CompressedState) -> Result<ReconstructedState, Box<dyn Error>> {
        // First validate the geometric proof
        if !self.validate_geometric_proof(&compressed.proof, &compressed.compressed_data)? {
            return Err("Invalid geometric proof for compressed state".into());
        }

        let mut reconstructed = Vec::new();
        let mut total_confidence = 0.0;

        // Extract points from compressed data
        let points = self.extract_points_from_compressed(&compressed.compressed_data)?;

        // Reconstruct state along geodesic path
        for window in compressed.geodesic_path.windows(2) {
            let start = window[0];
            let end = window[1];
            
            // Find points that should exist between these geodesic points
            let intermediate_points = self.reconstruct_intermediate_points(&start, &end, &points)?;
            
            // Calculate confidence for these reconstructed points
            let confidence = self.calculate_reconstruction_confidence(
                &intermediate_points,
                &compressed.proof.tessellation_points
            );
            
            reconstructed.extend(intermediate_points);
            total_confidence += confidence;
        }

        // Generate verification hash for reconstructed state
        let mut hasher = Sha256::new();
        for point in &reconstructed {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        let verification_hash = format!("{:x}", hasher.finalize());

        Ok(ReconstructedState {
            nodes: reconstructed,
            confidence: total_confidence / compressed.geodesic_path.len() as f64,
            verification_hash,
        })
    }

    fn validate_geometric_proof(&self, proof: &GeometricProof, compressed_data: &[u8]) -> Result<bool, Box<dyn Error>> {
        // Verify the proof hash
        let mut hasher = Sha256::new();
        for point in &proof.verification_path {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        hasher.update(compressed_data);
        let calculated_hash = format!("{:x}", hasher.finalize());

        if calculated_hash != proof.hash {
            return Ok(false);
        }

        // Verify tessellation coverage
        let coverage = self.verify_tessellation_coverage(&proof.tessellation_points, &proof.verification_path);
        if coverage < 0.9 { // Require 90% coverage
            return Ok(false);
        }

        Ok(true)
    }

    fn extract_points_from_compressed(&self, compressed_data: &[u8]) -> Result<Vec<Point>, Box<dyn Error>> {
        let mut points = Vec::new();
        
        for chunk in compressed_data.chunks(16) {
            if chunk.len() >= 16 {
                let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
                let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
                
                if x.is_finite() && y.is_finite() {
                    points.push(Point { x, y });
                }
            }
        }
        
        Ok(points)
    }

    fn reconstruct_intermediate_points(&self, start: &Point, end: &Point, reference_points: &[Point]) -> Result<Vec<Point>, Box<dyn Error>> {
        let mut points = Vec::new();
        let steps = (self.hyperbolic_distance(start, end) * 10.0).ceil() as i32;
        
        for i in 0..=steps {
            let t = i as f64 / steps as f64;
            let interpolated = self.interpolate_points(start, end, t);
            
            // Find closest reference point if any exists nearby
            if let Some(ref_point) = self.find_closest_reference_point(&interpolated, reference_points, 0.1) {
                points.push(ref_point);
            } else {
                points.push(interpolated);
            }
        }
        
        Ok(points)
    }

    fn find_closest_reference_point(&self, point: &Point, reference_points: &[Point], max_distance: f64) -> Option<Point> {
        reference_points.iter()
            .filter(|&p| self.hyperbolic_distance(point, p) <= max_distance)
            .min_by(|&a, &b| {
                let dist_a = self.hyperbolic_distance(point, a);
                let dist_b = self.hyperbolic_distance(point, b);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }

    fn calculate_reconstruction_confidence(&self, reconstructed: &[Point], tessellation: &[Point]) -> f64 {
        let mut total_confidence = 0.0;
        let mut points_verified = 0;

        for point in reconstructed {
            // Find closest tessellation points
            let mut distances: Vec<_> = tessellation.iter()
                .map(|t| self.hyperbolic_distance(point, t))
                .collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Calculate confidence based on proximity to tessellation points
            if let Some(&min_dist) = distances.first() {
                let confidence = (-min_dist).exp(); // Exponential decay with distance
                total_confidence += confidence;
                points_verified += 1;
            }
        }

        if points_verified == 0 {
            0.0
        } else {
            total_confidence / points_verified as f64
        }
    }

    fn verify_tessellation_coverage(&self, tessellation: &[Point], path: &[Point]) -> f64 {
        let mut covered_points = 0;
        let max_distance = 0.1; // Maximum distance for considering a point covered

        for path_point in path {
            if tessellation.iter().any(|t| self.hyperbolic_distance(path_point, t) <= max_distance) {
                covered_points += 1;
            }
        }

        covered_points as f64 / path.len() as f64
    }

    // Method to get a compressed state from the cache
    fn get_compressed_state(&self, key: &str) -> Option<CompressedState> {
        let mut cache = self.compressed_state_cache.lock().unwrap();
        cache.get(key).cloned()
    }

    // Method to add a compressed state to the cache
    fn cache_compressed_state(&self, key: String, state: CompressedState) {
        let mut cache = self.compressed_state_cache.lock().unwrap();
        cache.put(key, state);
    }

    // Add this method after the existing compress_state method
    fn compress_state_with_metrics(&mut self, state_points: &[Point]) -> Result<(CompressedState, CompressionStats), Box<dyn Error>> {
        let start_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        // Calculate original size (approximate using point data)
        let original_size = state_points.len() * std::mem::size_of::<Point>();
        
        // Generate state key for caching
        let mut hasher = Sha256::new();
        for point in state_points {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        let state_key = format!("state_{:x}", hasher.finalize());

        // Check cache first
        if let Some(cached_state) = self.get_compressed_state(&state_key) {
            let end_time = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_micros() as u64;
            
            let compressed_size = std::mem::size_of::<CompressedState>();
            
            // Update metrics
            self.metrics.states_compressed += 1;
            self.metrics.total_original_size += original_size as u64;
            self.metrics.total_compressed_size += compressed_size as u64;
            self.metrics.total_compression_time += end_time - start_time;

            let stats = CompressionStats {
                original_size: original_size as u64,
                compressed_size: compressed_size as u64,
                compression_ratio: original_size as f64 / compressed_size as f64,
                compression_time: end_time - start_time,
                tessellation_points: cached_state.geodesic_path.len(),
            };

            return Ok((cached_state, stats));
        }

        // Generate geodesic path and optimized tessellation
        let geodesic_path = self.calculate_compression_path(state_points);
        let tessellation = self.generate_optimized_tessellation(&geodesic_path);
        
        // Compress state along geodesic path
        let (compressed_data, ratio) = self.compress_along_geodesic(&geodesic_path)?;
        
        // Generate geometric proof
        let proof = self.generate_geometric_proof(&geodesic_path, &compressed_data)?;
        
        let compressed_state = CompressedState {
            geodesic_path,
            compressed_data,
            compression_ratio: ratio,
            proof,
        };

        let end_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_micros() as u64;
        
        let compressed_size = std::mem::size_of::<CompressedState>();
        
        // Update metrics
        self.metrics.states_compressed += 1;
        self.metrics.total_original_size += original_size as u64;
        self.metrics.total_compressed_size += compressed_size as u64;
        self.metrics.total_compression_time += end_time - start_time;

        // Cache the result
        self.cache_compressed_state(state_key, compressed_state.clone());

        let stats = CompressionStats {
            original_size: original_size as u64,
            compressed_size: compressed_size as u64,
            compression_ratio: ratio,
            compression_time: end_time - start_time,
            tessellation_points: tessellation.len(),
        };

        Ok((compressed_state, stats))
    }

    // Add these helper methods for metrics reporting
    pub fn get_compression_metrics(&self) -> CompressionMetricsReport {
        let total_states = self.metrics.states_compressed;
        let avg_ratio = if self.metrics.total_compressed_size > 0 {
            self.metrics.total_original_size as f64 / self.metrics.total_compressed_size as f64
        } else {
            0.0
        };
        let avg_time = if total_states > 0 {
            self.metrics.total_compression_time / total_states
        } else {
            0
        };
        let failure_rate = if total_states > 0 {
            self.metrics.compression_failures as f64 / total_states as f64
        } else {
            0.0
        };

        CompressionMetricsReport {
            total_states_compressed: total_states,
            average_compression_ratio: avg_ratio,
            average_compression_time: avg_time,
            failure_rate,
        }
    }

    // Add method for quantum-secure transaction broadcasting
    async fn broadcast_quantum_transaction(&mut self, transaction: Transaction) -> Result<(), Box<dyn Error>> {
        let sender_id = transaction.from.clone();
        
        if let Some(sender) = self.users.get(&sender_id) {
            // Serialize transaction for proof generation
            let transaction_data = bincode::serialize(&transaction)?;
            let position = self.get_transaction_position(&transaction);
            let proof = sender.quantum_crypto.generate_geometric_proof(&transaction_data, &position)?;
            
            // Serialize transaction and proof together
            let transaction_data = (transaction, proof);
            // Store or use transaction_bytes if needed
            let _transaction_bytes = bincode::serialize(&transaction_data)?;
            
            // Encrypt using ML-KEM
            let (ciphertext, _shared_key) = sender.quantum_crypto.encapsulate()?;
            
            let message = MeshMessage::EncryptedTransaction {
                ciphertext,
                sender_id: sender_id.clone(),
            };
            
            self.broadcast_message(&message).await?;
        }
        
        Ok(())
    }

    // Add method to handle encrypted messages
    async fn handle_encrypted_message(&mut self, ciphertext: Vec<u8>, sender_id: String) -> Result<(), Box<dyn Error>> {
        if let Some(receiver) = self.users.get(&self.local_peer_id.unwrap().to_string()) {
            match receiver.quantum_crypto.decapsulate(&ciphertext) {
                Ok(_shared_key) => {
                    println!("Successfully decapsulated message from {}", sender_id);
                }
                Err(e) => {
                    println!("Failed to decapsulate message: {:?}", e);
                }
            }
        }
        
        Ok(())
    }

    // Rename the async version
    async fn validate_transaction_proof(&self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        let transaction_data = bincode::serialize(&transaction)?;
        let position = self.get_transaction_position(transaction);
        let proof = self.users.get(&transaction.from)
            .ok_or("Sender not found")?
            .quantum_crypto
            .generate_geometric_proof(&transaction_data, &position)?;
        
        Ok(self.users.get(&transaction.from)
            .ok_or("Sender not found")?
            .quantum_crypto
            .verify_geometric_proof(&proof, &transaction_data)?)
    }

    async fn process_transaction(&mut self, transaction: Transaction) -> Result<(), Box<dyn Error>> {
        if self.validate_transaction(&transaction).await? {
            // Process the valid transaction
            if let Some(sender) = self.users.get_mut(&transaction.from) {
                sender.node.balance -= transaction.amount;
            }
            if let Some(receiver) = self.users.get_mut(&transaction.to) {
                receiver.node.balance += transaction.amount;
            }
        }
        Ok(())
    }

    async fn handle_transaction(&mut self, transaction: Transaction) -> Result<(), Box<dyn Error>> {
        if self.validate_transaction(&transaction).await? {
            self.process_transaction(transaction).await?;
        }
        Ok(())
    }

    async fn validate_and_process(&mut self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        if self.validate_transaction(transaction).await? {
            Ok(true)
        } else {
            Ok(false)
        }
    }

    async fn handle_mesh_message(&mut self, message: MeshMessage) -> Result<(), Box<dyn Error>> {
        match message {
            MeshMessage::Transaction(transaction) => {
                if self.validate_transaction(&transaction).await? {
                    self.process_transaction(transaction).await?;
                }
            },
            MeshMessage::Block(block) => {
                // Handle block message
                self.process_block(block).await?;
            },
            MeshMessage::RequestChain { from_block } => {
                // Handle chain request
                self.handle_chain_request(from_block).await?;
            },
            MeshMessage::ChainResponse { blocks } => {
                // Handle chain response
                self.process_chain_response(blocks).await?;
            },
            MeshMessage::MessageAck { message_id } => {
                // Handle message acknowledgment
                self.handle_message_ack(&message_id).await?;
            },
            MeshMessage::RetransmissionRequest { message_id } => {
                // Handle retransmission request
                self.handle_retransmission_request(&message_id).await?;
            },
            MeshMessage::QuantumKeyExchange { public_key: _, sender_id } => {
                // Handle quantum key exchange
                self.handle_quantum_key_exchange(&sender_id).await?;
            },
            _ => {
                // Handle any other message types
                println!("Unhandled message type");
            }
        }
        Ok(())
    }

    async fn process_pending_transactions(&mut self) -> Result<(), Box<dyn Error>> {
        while let Some(transaction) = self.pending_transactions.pop_front() {
            if self.validate_transaction(&transaction).await? {
                self.process_transaction(transaction).await?;
            }
        }
        Ok(())
    }

    async fn handle_transaction_sync(&mut self, transaction: &Transaction) -> Result<bool, Box<dyn Error>> {
        if self.validate_transaction(transaction).await? {
            self.pending_transactions.push_back(transaction.clone());
            Ok(true)
        } else {
            Ok(false)
        }
    }

    // Update the synchronous call to use tokio's block_on
    fn process_transaction_sync(&mut self, transaction: &Transaction) -> bool {
        tokio::runtime::Runtime::new()
            .unwrap()
            .block_on(self.handle_transaction_sync(transaction))
            .unwrap_or(false)
    }

    async fn process_block(&mut self, block: Block) -> Result<(), Box<dyn Error>> {
        if self.validate_propagating_block(&block).await? {
            self.blocks.push(block);
            // Process transactions in the block
            // Update balances, etc.
        }
        Ok(())
    }

    async fn handle_chain_request(&mut self, from_block: usize) -> Result<(), Box<dyn Error>> {
        if let Some(blocks) = self.get_blocks_from(from_block) {
            let response = MeshMessage::ChainResponse { blocks };
            self.broadcast_message(&response).await?;
        }
        Ok(())
    }

    async fn handle_message_ack(&mut self, message_id: &str) -> Result<(), Box<dyn Error>> {
        self.pending_messages.remove(message_id);
        Ok(())
    }

    async fn handle_retransmission_request(&mut self, message_id: &str) -> Result<(), Box<dyn Error>> {
        // Clone the message first to avoid borrow checker issues
        let message_to_broadcast = if let Some((message, _)) = self.pending_messages.get(message_id) {
            message.clone()
        } else {
            return Ok(());
        };
        
        self.broadcast_message(&message_to_broadcast).await?;
        Ok(())
    }

    async fn handle_quantum_key_exchange(&mut self, sender_id: &str) -> Result<(), Box<dyn Error>> {
        if let Some(_sender) = self.users.get(sender_id) {
            // Process quantum key exchange
            println!("Processing quantum key exchange from {}", sender_id);
        }
        Ok(())
    }

    // Add these new methods for consensus

    /// Generates a consensus proof for a block using geometric properties
    async fn generate_consensus_proof(&self, block: &Block) -> Result<GeometricProof, Box<dyn Error>> {
        // Serialize block data for proof generation
        let block_data = bincode::serialize(&block)?;
        
        // Calculate the block's position in hyperbolic space based on its transactions
        let position = self.calculate_block_position(block);
        
        // Get the optimal consensus path through validator nodes
        let consensus_path = self.calculate_consensus_path(&position);
        
        // Generate tessellation points for validation
        let tessellation = self.generate_optimized_tessellation(&consensus_path);
        
        // Calculate verification path through validators
        let verification_path = self.calculate_verification_path(&tessellation, &block_data);
        
        // Generate proof hash incorporating consensus path
        let mut hasher = Sha256::new();
        for point in &verification_path {
            hasher.update(format!("{}{}", point.x, point.y));
        }
        hasher.update(&block_data);
        let proof_hash = format!("{:x}", hasher.finalize());
        
        // Sign the proof using quantum-resistant signatures
        let signature = if let Some(local_user) = self.users.get(&self.local_peer_id.expect("No local peer ID").to_string()) {
            local_user.quantum_crypto.sign_transaction(&block_data)?
        } else {
            return Err("Local user not found".into());
        };
        
        Ok(GeometricProof {
            tessellation_points: tessellation,
            verification_path,
            position_data: bincode::serialize(&position)?,
            hash: proof_hash,
            signature,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_secs(),
        })
    }

    /// Validates a consensus proof for a block
    async fn validate_consensus_proof(&self, block: &Block, proof: &GeometricProof) -> Result<bool, Box<dyn Error>> {
        // Verify the basic geometric proof
        let block_data = bincode::serialize(&block)?;
        let mut hasher = Sha256::new();
        hasher.update(&block_data);
        let calculated_hash = format!("{:x}", hasher.finalize());
        
        if calculated_hash != proof.hash {
            return Ok(false);
        }
        
        // Verify the consensus path
        let position: Point = bincode::deserialize(&proof.position_data)?;
        if !self.verify_consensus_path(&proof.verification_path, &position) {
            return Ok(false);
        }
        
        // Verify tessellation coverage
        let coverage = self.verify_tessellation_coverage(&proof.tessellation_points, &proof.verification_path);
        if coverage < 0.9 { // Require 90% coverage
            return Ok(false);
        }
        
        // Additional consensus-specific validation using block points
        let block_points = self.block_to_points(block)?;
        if !self.validate_consensus_distance(&block_points, proof)? {
            return Ok(false);
        }
        
        // Verify quantum-resistant signature
        if let Some(validator) = self.find_block_validator(block) {
            let is_valid = validator.quantum_crypto.verify_signature(
                &block_data,
                &proof.signature,
                validator.quantum_crypto.get_signing_public_key()
            )?;
            if !is_valid {
                return Ok(false);
            }
        } else {
            return Ok(false);
        }
        
        Ok(true)
    }

    /// Calculates the optimal consensus path through validator nodes
    fn calculate_consensus_path(&self, position: &Point) -> Vec<Point> {
        let mut path = Vec::new();
        let mut current = Point { x: 0.0, y: 0.0 }; // Start at origin
        path.push(current);
        
        // Get validator nodes sorted by hyperbolic distance
        let mut validators: Vec<_> = self.users.values()
            .filter(|user| user.node.is_validator)
            .collect();
        
        validators.sort_by(|a, b| {
            let dist_a = self.hyperbolic_distance(&a.node.position, position);
            let dist_b = self.hyperbolic_distance(&b.node.position, position);
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Build path through validators
        for validator in validators.iter().take(3) { // Use top 3 closest validators
            let validator_pos = validator.node.position;
            
            // Add intermediate points for continuous path
            let steps = (self.hyperbolic_distance(&current, &validator_pos) * 10.0).ceil() as i32;
            for i in 1..=steps {
                let t = i as f64 / steps as f64;
                let interpolated = self.interpolate_points(&current, &validator_pos, t);
                path.push(interpolated);
            }
            
            current = validator_pos;
        }
        
        // Add final path to target position
        let steps = (self.hyperbolic_distance(&current, position) * 10.0).ceil() as i32;
        for i in 1..=steps {
            let t = i as f64 / steps as f64;
            let interpolated = self.interpolate_points(&current, position, t);
            path.push(interpolated);
        }
        
        path
    }

    /// Verifies that a consensus path is valid
    fn verify_consensus_path(&self, path: &[Point], target: &Point) -> bool {
        // Check if path starts near origin
        if !path.is_empty() && self.hyperbolic_distance(&path[0], &Point { x: 0.0, y: 0.0 }) > 0.1 {
            return false;
        }
        
        // Check path continuity
        for window in path.windows(2) {
            let distance = self.hyperbolic_distance(&window[0], &window[1]);
            if distance > 0.2 { // Maximum allowed gap in path
                return false;
            }
        }
        
        // Check if path reaches target
        if let Some(last) = path.last() {
            let final_distance = self.hyperbolic_distance(last, target);
            if final_distance > 0.1 { // Maximum allowed distance from target
                return false;
            }
        } else {
            return false;
        }
        
        // Verify path passes through sufficient validators
        let validator_count = path.iter()
            .filter(|&point| {
                self.users.values()
                    .any(|user| {
                        user.node.is_validator && 
                        self.hyperbolic_distance(&user.node.position, point) < 0.1
                    })
            })
            .count();
        
        validator_count >= 3 // Require at least 3 validators
    }

    /// Calculates the position of a block in hyperbolic space
    fn calculate_block_position(&self, block: &Block) -> Point {
        // Calculate weighted average of transaction positions
        let mut total_x = 0.0;
        let mut total_y = 0.0;
        let mut total_weight = 0.0;
        
        for tx in &block.transactions {
            let pos = self.get_transaction_position(tx);
            let weight = tx.amount;
            
            total_x += pos.x * weight;
            total_y += pos.y * weight;
            total_weight += weight;
        }
        
        if total_weight > 0.0 {
            Point {
                x: total_x / total_weight,
                y: total_y / total_weight,
            }
        } else {
            Point { x: 0.0, y: 0.0 }
        }
    }

    /// Finds the validator responsible for a block
    fn find_block_validator(&self, block: &Block) -> Option<&User> {
        let position = self.calculate_block_position(block);
        
        self.users.values()
            .filter(|user| user.node.is_validator)
            .min_by(|a, b| {
                let dist_a = self.hyperbolic_distance(&a.node.position, &position);
                let dist_b = self.hyperbolic_distance(&b.node.position, &position);
                dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
            })
    }

    // Add method to designate a node as validator
    fn set_node_as_validator(&mut self, node_id: &str, is_validator: bool) -> Result<(), Box<dyn Error>> {
        if let Some(user) = self.users.get_mut(node_id) {
            user.node.is_validator = is_validator;
            Ok(())
        } else {
            Err("Node not found".into())
        }
    }

    fn validate_consensus_distance(&self, points: &[Point], proof: &GeometricProof) -> Result<bool, Box<dyn Error>> {
        // Calculate the average distance between all points and verification path
        let mut total_distance = 0.0;
        let mut count = 0;
        
        // Get validator positions from verification path
        for point in points {
            for vp_point in &proof.verification_path {
                // Use the hyperbolic distance calculation
                let distance = hyperbolic_distance(point, vp_point);
                total_distance += distance;
                count += 1;
            }
        }
        
        if count == 0 {
            return Ok(false); // No valid points to verify
        }
        
        let average_distance = total_distance / count as f64;
        
        // Check if we have enough validator coverage
        let validator_count = proof.verification_path.iter()
            .filter(|&point| self.is_validator_position(point))
            .count();
            
        // Consensus requires both:
        // 1. Average distance below threshold
        // 2. Sufficient validator coverage (at least 3 validators)
        Ok(average_distance < 1.0 && validator_count >= 3)
    }

    fn is_validator_position(&self, point: &Point) -> bool {
        self.users.values()
            .any(|user| {
                user.node.is_validator && 
                hyperbolic_distance(&user.node.position, point) < 0.1
            })
    }

    fn block_to_points(&self, block: &Block) -> Result<Vec<Point>, Box<dyn Error>> {
        let mut points = Vec::new();
        
        // Add block's position based on its transactions
        let block_position = self.calculate_block_position(block);
        points.push(block_position);
        
        // Add points for each transaction in the block
        for transaction in &block.transactions {
            // Get sender and receiver positions, skipping if not found
            if let Some(sender) = self.users.get(&transaction.from) {
                points.push(sender.node.position);
                
                if let Some(receiver) = self.users.get(&transaction.to) {
                    points.push(receiver.node.position);
                    
                    // Calculate transaction point only if both sender and receiver exist
                    let transaction_point = self.calculate_transaction_point(
                        &sender.node.position,
                        &receiver.node.position,
                        transaction.amount
                    );
                    points.push(transaction_point);
                }
            }
        }
        
        // Add validator positions
        for user in self.users.values() {
            if user.node.is_validator {
                points.push(user.node.position);
            }
        }
        
        if points.is_empty() {
            return Err("No valid points found for block".into());
        }
        
        Ok(points)
    }

    fn calculate_transaction_point(&self, sender: &Point, receiver: &Point, amount: f64) -> Point {
        // Calculate a point along the geodesic path between sender and receiver
        // weighted by the transaction amount
        let weight = amount.min(1.0).max(0.0); // Normalize amount to [0,1]
        Point {
            x: sender.x + (receiver.x - sender.x) * weight,
            y: sender.y + (receiver.y - sender.y) * weight,
        }
    }
}

// Add this struct for metrics reporting
#[derive(Debug)]
pub struct CompressionMetricsReport {
    pub total_states_compressed: u64,
    pub average_compression_ratio: f64,
    pub average_compression_time: u64,
    pub failure_rate: f64,
}

// Add this struct for compression statistics
#[derive(Debug)]
pub struct CompressionStats {
    pub original_size: u64,
    pub compressed_size: u64,
    pub compression_ratio: f64,
    pub compression_time: u64,
    pub tessellation_points: usize,
}


impl MerkleTree {
    fn new(transactions: &[Transaction]) -> Self {
        let mut leaves: Vec<String> = transactions
            .iter()
            .map(|tx| Self::hash_transaction(tx))
            .collect();

        // Ensure even number of leaves by duplicating last one if necessary
        if leaves.len() % 2 == 1 {
            leaves.push(leaves.last().unwrap().clone());
        }

        let root = Self::build_tree(&leaves);

        MerkleTree {
            root,
            leaves,
        }
    }

    fn hash_transaction(transaction: &Transaction) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}{}{}", 
            transaction.from,
            transaction.to,
            transaction.amount,
            transaction.timestamp
        ));
        format!("{:x}", hasher.finalize())
    }

    fn hash_pair(left: &str, right: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(format!("{}{}", left, right));
        format!("{:x}", hasher.finalize())
    }

    fn build_tree(leaves: &[String]) -> String {
        if leaves.is_empty() {
            return "".to_string();
        }
        if leaves.len() == 1 {
            return leaves[0].clone();
        }

        let mut next_level = Vec::new();
        for chunk in leaves.chunks(2) {
            let hash = if chunk.len() == 2 {
                Self::hash_pair(&chunk[0], &chunk[1])
            } else {
                Self::hash_pair(&chunk[0], &chunk[0])
            };
            next_level.push(hash);
        }

        Self::build_tree(&next_level)
    }

    fn verify_transaction(&self, transaction: &Transaction) -> bool {
        let tx_hash = Self::hash_transaction(transaction);
        self.verify_proof(&tx_hash)
    }

    fn verify_proof(&self, leaf_hash: &str) -> bool {
        self.leaves.contains(&leaf_hash.to_string())
    }
}

#[derive(Clone, Debug)]
struct SerializableKeypair {
    bytes: Vec<u8>,
}

impl Serialize for SerializableKeypair {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_bytes(&self.bytes)
    }
}

impl<'de> Deserialize<'de> for SerializableKeypair {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        struct BytesVisitor;

        impl<'de> serde::de::Visitor<'de> for BytesVisitor {
            type Value = SerializableKeypair;

            fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
                formatter.write_str("a byte array")
            }

            fn visit_bytes<E>(self, v: &[u8]) -> Result<Self::Value, E>
            where
                E: serde::de::Error,
            {
                Ok(SerializableKeypair {
                    bytes: v.to_vec(),
                })
            }
        }

        deserializer.deserialize_bytes(BytesVisitor)
    }
}

impl HyperbolicRegion {
    fn new(center: Point, radius: f64) -> Self {
         // Calculate boundary points first while we can still borrow center
         let boundary_points = Self::calculate_boundary_points(&center, radius);
        
        HyperbolicRegion {
            center,
            radius,
            nodes: Vec::new(),
            boundary_points,
        }
    }

    fn calculate_boundary_points(center: &Point, radius: f64) -> Vec<Point> {
        let num_points = 16; // Number of points to approximate the boundary
        let mut boundary = Vec::with_capacity(num_points);
        
        for i in 0..num_points {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            let x = center.x + radius * angle.cos();
            let y = center.y + radius * angle.sin();
            
            // Project point onto Poincaré disk if necessary
            let norm = (x * x + y * y).sqrt();
            if norm > 1.0 {
                let scale = 0.99 / norm;
                boundary.push(Point { x: x * scale, y: y * scale });
            } else {
                boundary.push(Point { x, y });
            }
        }
        
        boundary
    }

    fn contains_point(&self, point: &Point, mesh: &QuantumMesh) -> bool {
        let distance = mesh.hyperbolic_distance(&self.center, point);
        distance <= self.radius
    }

    fn calculate_metrics_from_users(&self, users: &HashMap<String, User>) -> RegionMetrics {
        let mut total_transactions = 0;
        let mut total_load = 0.0;

        for node_id in &self.nodes {
            if let Some(user) = users.get(node_id) {
                total_transactions += user.node.transactions.len();
                // Calculate load based on transaction history and connections
                let node_load = user.node.transactions.len() as f64 * 
                               user.node.connections.len() as f64;
                total_load += node_load;
            }
        }

        RegionMetrics {
            transaction_density: total_transactions as f64 / (std::f64::consts::PI * self.radius * self.radius),
            node_count: self.nodes.len(),
            average_load: if self.nodes.is_empty() { 0.0 } else { total_load / self.nodes.len() as f64 },
            boundary_crossings: 0,
        }
    }
}

impl HyperbolicRouter {
  

    fn cluster_nodes(&self, users: &HashMap<String, User>) -> Vec<Vec<String>> {
        let mut clusters = Vec::new();
        let mut processed_nodes = HashSet::new();
        
        for (node_id, user) in users {
            if processed_nodes.contains(node_id) {
                continue;
            }
            
            let mut cluster = Vec::new();
            let mut queue = VecDeque::new();
            queue.push_back(node_id.clone());
            
            while let Some(current_id) = queue.pop_front() {
                if processed_nodes.contains(&current_id) {
                    continue;
                }
                
                processed_nodes.insert(current_id.clone());
                cluster.push(current_id.clone());
                
                // Add neighbors to queue
                if let Some(neighbors) = self.routing_table.get(&current_id) {
                    for neighbor in neighbors {
                        if !processed_nodes.contains(neighbor) {
                            queue.push_back(neighbor.clone());
                        }
                    }
                }
            }
            
            if !cluster.is_empty() {
                clusters.push(cluster);
            }
        }
        
        clusters
    }

    fn calculate_region_parameters(&self, cluster: &[String], users: &HashMap<String, User>) -> (Point, f64) {
        // Calculate center as weighted average of node positions
        let mut center_x = 0.0;
        let mut center_y = 0.0;
        let mut total_weight = 0.0;
        
        for node_id in cluster {
            if let Some(user) = users.get(node_id) {
                let weight = user.node.transactions.len() as f64 + 1.0;
                center_x += user.node.position.x * weight;
                center_y += user.node.position.y * weight;
                total_weight += weight;
            }
        }
        
        let center = if total_weight > 0.0 {
            Point {
                x: center_x / total_weight,
                y: center_y / total_weight,
            }
        } else {
            Point { x: 0.0, y: 0.0 }
        };
        
        // Calculate radius to encompass all nodes with some padding
        let mut max_distance: f64 = 0.0;
        for node_id in cluster {
            if let Some(user) = users.get(node_id) {
                let distance = hyperbolic_distance(&center, &user.node.position);
                max_distance = f64::max(max_distance, distance);
            }
        }
        
        // Add 20% padding to the radius
        let radius = max_distance * 1.2;
        
        (center, radius)
    }
}




#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Get username from command line args or environment
    let username = std::env::args()
        .nth(1)
        .expect("Please provide a username");

    let mut mesh = QuantumMesh::new(2, 1000);

    // Register and start a single user node
    mesh.register_user(username.clone()).await?;
    mesh.start_user_node(&username).await?;

    Ok(())
}