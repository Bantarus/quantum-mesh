use crate::types::{User, HyperbolicRegion, Block};
use crate::quantum_crypto::{QuantumCrypto, Point, GeometricProof};
use sha2::{Sha256, Digest};
use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use bincode;
use std::collections::HashSet;
use rand::{Rng, thread_rng};
use crate::hyperbolic::hyperbolic_distance;

pub struct GenesisConfig {
    pub initial_validators: Vec<User>,
    pub initial_regions: Vec<HyperbolicRegion>,
    pub network_parameters: NetworkParameters,
    pub timestamp: u64,
}

#[derive(Clone)]
pub struct NetworkParameters {
    pub max_validators: usize,
    pub min_stake: f64,
    pub initial_supply: f64,
    pub region_count: usize,
    pub tessellation_density: f64,
}

pub struct GenesisBlock {
    pub block: Block,
    pub proof: GeometricProof,
    pub initial_state: GenesisState,
}

#[derive(Clone)]
pub struct GenesisState {
    pub validator_positions: Vec<(String, Point)>,
    pub region_boundaries: Vec<HyperbolicRegion>,
    pub initial_tessellation: Vec<Point>,
}

impl GenesisBlock {
    pub async fn create(config: GenesisConfig) -> Result<Self, Box<dyn Error>> {
        // 1. Initialize quantum-resistant cryptography for genesis
        let genesis_crypto = QuantumCrypto::new()?;
        
        // 2. Generate initial tessellation for the hyperbolic space
        let initial_tessellation = Self::generate_initial_tessellation(
            &config.network_parameters.tessellation_density
        )?;
        
        // 3. Position initial validators in hyperbolic space
        let validator_positions = Self::position_validators(&config.initial_validators)?;
        
        // 4. Create initial regions
        let region_boundaries = Self::create_initial_regions(
            &validator_positions,
            config.network_parameters.region_count
        )?;
        
        // 5. Create genesis state
        let genesis_state = GenesisState {
            validator_positions: validator_positions.clone(),
            region_boundaries: region_boundaries.clone(),
            initial_tessellation,
        };
        
        // 6. Create genesis block
        let block = Block {
            previous_hash: "0000000000000000000000000000000000000000000000000000000000000000".to_string(),
            timestamp: config.timestamp,
            transactions: vec![], // Genesis block has no transactions
            hash: Self::calculate_genesis_hash(&genesis_state)?,
            height: 0,
          //  difficulty: 0,       // Genesis block difficulty is typically 0
         //   nonce: 0,            // No mining needed for genesis
        //    merkle_root: "0".to_string(), // No transactions means empty merkle root
        };
        
        // 7. Generate geometric proof for genesis block
        let proof = Self::generate_genesis_proof(&block, &genesis_state, &genesis_crypto)?;
        
        Ok(GenesisBlock {
            block,
            proof,
            initial_state: genesis_state,
        })
    }

    fn generate_initial_tessellation(density: &f64) -> Result<Vec<Point>, Box<dyn Error>> {
        let mut tessellation = Vec::new();
        let mut rng = thread_rng();
        
        // Number of layers based on density
        let layers = (1.0 / density).ceil() as i32;
        
        // First add the origin
        tessellation.push(Point { x: 0.0, y: 0.0 });
        
        // Generate concentric layers of tessellation with adaptive density
        for layer in 1..=layers {
            let radius = density * layer as f64;
            // Number of points increases with the hyperbolic circumference
            let points_in_layer = (2.0 * std::f64::consts::PI * layer as f64 / density) as i32;
            
            for i in 0..points_in_layer {
                let angle = 2.0 * std::f64::consts::PI * (i as f64) / (points_in_layer as f64);
                
                // Calculate hyperbolic position using the Poincaré disk model
                let tanh_r = radius.tanh();
                let x = tanh_r * angle.cos();
                let y = tanh_r * angle.sin();
                
                // Ensure point is within the Poincaré disk
                if x * x + y * y < 1.0 {
                    tessellation.push(Point { x, y });
                }
            }
        }
        
        // Add random jitter to break symmetry (important for robust verification)
        for point in &mut tessellation {
            if point.x != 0.0 || point.y != 0.0 {  // Don't move the origin
                let jitter = density * 0.1;  // 10% of density
                point.x += rng.gen_range(-jitter..jitter);
                point.y += rng.gen_range(-jitter..jitter);
                
                // Ensure we stay in the disk
                let norm = (point.x * point.x + point.y * point.y).sqrt();
                if norm >= 0.99 {  // Leave a small margin
                    let scale = 0.98 / norm;  // Scale back slightly
                    point.x *= scale;
                    point.y *= scale;
                }
            }
        }
        
        Ok(tessellation)
    }

    fn position_validators(
        validators: &[User]
    ) -> Result<Vec<(String, Point)>, Box<dyn Error>> {
        let mut positions = Vec::new();
        let min_hyperbolic_distance = 0.5;  // Minimum separation in hyperbolic space
        let mut rng = thread_rng();
        
        // Position validators with minimum separation
        for validator in validators {
            let mut position_found = false;
            let mut attempts = 0;
            
            while !position_found && attempts < 100 {
                // Generate candidate position using the Poincaré disk model
                // We use a factor of 0.9 to keep validators away from the boundary
                let r = 0.9 * (rng.gen::<f64>()).sqrt();  // Square root for uniform disk distribution
                let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                
                let candidate = Point {
                    x: r * theta.cos(),
                    y: r * theta.sin(),
                };
                
                // Check if this position maintains minimum hyperbolic distance
                let mut valid_position = true;
                for (_, existing) in &positions {
                    let distance = hyperbolic_distance(&candidate, existing);
                    if distance < min_hyperbolic_distance {
                        valid_position = false;
                        break;
                    }
                }
                
                if valid_position {
                    position_found = true;
                    positions.push((validator.id.clone(), candidate));
                }
                
                attempts += 1;
            }
            
            // If we couldn't find a good position after max attempts, use the last candidate
            if !position_found {
                let r = 0.9 * (rng.gen::<f64>()).sqrt();
                let theta = rng.gen::<f64>() * 2.0 * std::f64::consts::PI;
                
                positions.push((
                    validator.id.clone(),
                    Point {
                        x: r * theta.cos(),
                        y: r * theta.sin(),
                    }
                ));
            }
        }
        
        Ok(positions)
    }

    fn create_initial_regions(
        validator_positions: &[(String, Point)],
        region_count: usize
    ) -> Result<Vec<HyperbolicRegion>, Box<dyn Error>> {
        if validator_positions.is_empty() {
            return Ok(vec![]);
        }
        
        // Use k-means clustering to create initial regions
        let mut regions = Vec::new();
        
        // Initially distribute region centers uniformly
        let mut region_centers = Vec::with_capacity(region_count);
        for i in 0..region_count {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (region_count as f64);
            let radius = 0.5;  // Middle distance in the disk
            
            region_centers.push(Point {
                x: radius * angle.cos(),
                y: radius * angle.sin(),
            });
        }
        
        // Assign validators to nearest region and update centers
        for _ in 0..10 {  // Run a few k-means iterations
            // Create empty assignments
            let mut region_assignments: Vec<Vec<usize>> = vec![Vec::new(); region_count];
            
            // Assign each validator to nearest region center
            for (i, (_, position)) in validator_positions.iter().enumerate() {
                let mut min_distance = f64::MAX;
                let mut nearest_region = 0;
                
                for (j, center) in region_centers.iter().enumerate() {
                    let distance = hyperbolic_distance(position, center);
                    if distance < min_distance {
                        min_distance = distance;
                        nearest_region = j;
                    }
                }
                
                region_assignments[nearest_region].push(i);
            }
            
            // Update region centers
            for (i, assignments) in region_assignments.iter().enumerate() {
                if !assignments.is_empty() {
                    let mut x_sum = 0.0;
                    let mut y_sum = 0.0;
                    
                    for &idx in assignments {
                        x_sum += validator_positions[idx].1.x;
                        y_sum += validator_positions[idx].1.y;
                    }
                    
                    let new_x = x_sum / assignments.len() as f64;
                    let new_y = y_sum / assignments.len() as f64;
                    
                    // Ensure center is within the Poincaré disk
                    let norm = (new_x * new_x + new_y * new_y).sqrt();
                    if norm < 0.99 {
                        region_centers[i] = Point { x: new_x, y: new_y };
                    } else {
                        // Scale back to stay within disk
                        let scale = 0.98 / norm;
                        region_centers[i] = Point { 
                            x: new_x * scale, 
                            y: new_y * scale 
                        };
                    }
                }
            }
        }
        
        // Create hyperbolic regions with appropriate radii
        for (i, center) in region_centers.iter().enumerate() {
            // Calculate maximum distance to any assigned validator
            let mut max_distance: f64 = 0.1;  // Minimum radius
            
            for (_, position) in validator_positions {
                let distance = hyperbolic_distance(center, position);
                max_distance = max_distance.max(distance);
            }
            
            // Add 20% margin to radius
            let radius = max_distance * 1.2;
            
            // Create region with calculated center and radius
            let boundary_points = Self::calculate_boundary_points(center, radius);
            
            let region = HyperbolicRegion {
                center: center.clone(),
                radius,
                nodes: Vec::new(),  // Will be populated later during node assignment
                boundary_points,
            };
            
            regions.push(region);
        }
        
        Ok(regions)
    }
    
    fn calculate_boundary_points(center: &Point, radius: f64) -> Vec<Point> {
        let num_points = 16;  // Number of points to approximate the boundary
        let mut boundary = Vec::with_capacity(num_points);
        
        for i in 0..num_points {
            let angle = 2.0 * std::f64::consts::PI * (i as f64) / (num_points as f64);
            
            // Use proper hyperbolic math for boundary points
            let tanh_r = radius.tanh();
            let dx = tanh_r * angle.cos();
            let dy = tanh_r * angle.sin();
            
            // Apply Möbius transformation to center at given point
            let denom = 1.0 + 2.0 * (center.x * dx + center.y * dy) + (dx * dx + dy * dy);
            let x = (dx + center.x) / denom;
            let y = (dy + center.y) / denom;
            
            // Ensure point is within Poincaré disk
            let norm = (x * x + y * y).sqrt();
            if norm > 0.99 {
                let scale = 0.98 / norm;
                boundary.push(Point { x: x * scale, y: y * scale });
            } else {
                boundary.push(Point { x, y });
            }
        }
        
        boundary
    }

    fn calculate_genesis_hash(state: &GenesisState) -> Result<String, Box<dyn Error>> {
        let mut hasher = Sha256::new();
        
        // Hash validator positions
        for (id, point) in &state.validator_positions {
            hasher.update(id.as_bytes());
            hasher.update(&point.x.to_le_bytes());
            hasher.update(&point.y.to_le_bytes());
        }
        
        // Hash region boundaries
        for region in &state.region_boundaries {
            hasher.update(&region.center.x.to_le_bytes());
            hasher.update(&region.center.y.to_le_bytes());
            hasher.update(&region.radius.to_le_bytes());
            
            for point in &region.boundary_points {
                hasher.update(&point.x.to_le_bytes());
                hasher.update(&point.y.to_le_bytes());
            }
        }
        
        // Hash initial tessellation
        for point in &state.initial_tessellation {
            hasher.update(&point.x.to_le_bytes());
            hasher.update(&point.y.to_le_bytes());
        }
        
        Ok(format!("{:x}", hasher.finalize()))
    }

    fn generate_genesis_proof(
        block: &Block,
        state: &GenesisState,
        crypto: &QuantumCrypto
    ) -> Result<GeometricProof, Box<dyn Error>> {
        // Serialize block and state together for proof
        let mut block_data = bincode::serialize(&block)?;
        
        // Add hash of initial state
        let state_hash = Self::calculate_genesis_hash(state)?;
        block_data.extend_from_slice(state_hash.as_bytes());
        
        // Generate geometric proof centered at origin
        let mut proof = crypto.generate_geometric_proof(&block_data, &Point { x: 0.0, y: 0.0 })?;
        
        // Add tessellation points from the initial state
        proof.tessellation_points = state.initial_tessellation.clone();
        
        // Generate verification path through validators and tessellation points
        proof.verification_path = Self::calculate_genesis_verification_path(state)?;
        
        Ok(proof)
    }
    
    fn calculate_genesis_verification_path(state: &GenesisState) -> Result<Vec<Point>, Box<dyn Error>> {
        let mut path = Vec::new();
        
        // Start at origin
        path.push(Point { x: 0.0, y: 0.0 });
        
        // Add paths to validators
        let mut visited = HashSet::new();
        
        // Sort validators by distance from origin for deterministic path
        let mut validators: Vec<_> = state.validator_positions.iter().collect();
        validators.sort_by(|(_, a), (_, b)| {
            let dist_a = a.x * a.x + a.y * a.y;
            let dist_b = b.x * b.x + b.y * b.y;
            dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Build paths to each validator through tessellation points
        for (_, validator_pos) in validators {
            if visited.contains(validator_pos) {
                continue;
            }
            
            // Find a path through tessellation
            let mut current = path.last().unwrap().clone();
            
            // Find tessellation points that form a path to the validator
            let mut candidates: Vec<_> = state.initial_tessellation.iter()
                .filter(|&p| !visited.contains(p))
                .collect();
            
            while hyperbolic_distance(&current, validator_pos) > 0.1 {
                // Sort by combined distance (to current + to target)
                candidates.sort_by(|&a, &b| {
                    let dist_a = hyperbolic_distance(&current, a) +
                                hyperbolic_distance(a, validator_pos);
                    let dist_b = hyperbolic_distance(&current, b) +
                                hyperbolic_distance(b, validator_pos);
                    dist_a.partial_cmp(&dist_b).unwrap_or(std::cmp::Ordering::Equal)
                });
                
                // Use closest point that's closer to target than current
                let mut found = false;
                for &candidate in &candidates {
                    if hyperbolic_distance(candidate, validator_pos) < 
                       hyperbolic_distance(&current, validator_pos) {
                        path.push(candidate.clone());
                        visited.insert(candidate.clone());
                        current = candidate.clone();
                        found = true;
                        break;
                    }
                }
                
                // If no good tessellation point, directly add validator
                if !found {
                    break;
                }
                
                // Update candidate list
                candidates = state.initial_tessellation.iter()
                    .filter(|&p| !visited.contains(p))
                    .collect();
            }
            
            // Add validator to path
            path.push(validator_pos.clone());
            visited.insert(validator_pos.clone());
        }
        
        Ok(path)
    }
}

// Add validation methods
impl GenesisBlock {
    pub fn validate(&self) -> Result<bool, Box<dyn Error>> {
        // 1. Verify block hash
        let calculated_hash = Self::calculate_genesis_hash(&self.initial_state)?;
        if calculated_hash != self.block.hash {
            return Ok(false);
        }
        
        // 2. Verify geometric proof
        let block_data = bincode::serialize(&self.block)?;
        let crypto = QuantumCrypto::new()?;
        let proof_valid = crypto.verify_geometric_proof(&self.proof, &block_data)?;
        if !proof_valid {
            return Ok(false);
        }
        
        // 3. Verify validator positions
        for (_, pos) in &self.initial_state.validator_positions {
            // Ensure validators are within the Poincaré disk
            let norm = (pos.x * pos.x + pos.y * pos.y).sqrt();
            if norm >= 1.0 {
                return Ok(false);
            }
        }
        
        // 4. Verify region distribution
        for region in &self.initial_state.region_boundaries {
            // Ensure region center is within disk
            let center_norm = (region.center.x * region.center.x + region.center.y * region.center.y).sqrt();
            if center_norm >= 1.0 {
                return Ok(false);
            }
            
            // Ensure region doesn't exceed disk
            let boundary_outside = region.boundary_points.iter().any(|p| {
                (p.x * p.x + p.y * p.y) >= 1.0
            });
            
            if boundary_outside {
                return Ok(false);
            }
        }
        
        // 5. Verify tessellation coverage
        let tessellation_covered = self.verify_tessellation_coverage()?;
        if !tessellation_covered {
            return Ok(false);
        }
        
        Ok(true)
    }
    
    fn verify_tessellation_coverage(&self) -> Result<bool, Box<dyn Error>> {
        // Check that tessellation has good coverage of validators
        for (_, pos) in &self.initial_state.validator_positions {
            // Find closest tessellation point
            let min_distance = self.initial_state.initial_tessellation.iter()
                .map(|p| hyperbolic_distance(p, pos))
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(f64::MAX);
                
            // Ensure at least one tessellation point is close enough
            if min_distance > 0.5 {
                return Ok(false);
            }
        }
        
        // Check that verification path is continuous
        for window in self.proof.verification_path.windows(2) {
            let distance = hyperbolic_distance(&window[0], &window[1]);
            if distance > 0.5 {
                return Ok(false);
            }
        }
        
        Ok(true)
    }
}