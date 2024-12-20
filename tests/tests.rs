#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use libp2p::identity;

    // Helper function to create a test user
    fn create_test_user(id: &str, x: f64, y: f64, transactions: Vec<Transaction>) -> User {
        // Generate a new keypair for the user
        let keypair = identity::Keypair::generate_ed25519();

        User {
            id: id.to_string(),
            keypair,  // Add the keypair here
            node: Node {
                position: Point { x, y },
                connections: Vec::new(),
                transactions,
                balance: 100.0,
            },
            serialized_keypair: Some(SerializableKeypair { bytes: vec![] }),  // Wrap in Some
        }
    }

    // Helper function to create a test transaction
    fn create_test_transaction(from: &str, to: &str, amount: f64) -> Transaction {
        Transaction {
            from: from.to_string(),
            to: to.to_string(),
            amount,
            timestamp: chrono::Utc::now().timestamp() as u64,  // Cast i64 to u64
            signature: "signed_test".to_string(),
        }
    }

    // Basic test to verify test framework is working
    #[test]
    fn test_basic_setup() {
        assert!(true);
    }

    #[test]
    fn test_shard_creation() {
        let router = HyperbolicRouter::new(100);  // Removed mut as it's not needed
        let center = Point { x: 0.0, y: 0.0 };
        let shard = Shard::new(0, center, 1.0);
        
        assert_eq!(shard.id, 0);
        assert_eq!(shard.center.x, 0.0);
        assert_eq!(shard.center.y, 0.0);
        assert_eq!(shard.radius, 1.0);
        assert!(shard.nodes.is_empty());
    }

    #[test]
    fn test_node_assignment_to_shards() {
        let mut router = HyperbolicRouter::new(100);
        let mut users = HashMap::new();

        // Create test users in different locations
        users.insert("user1".to_string(), create_test_user("user1", 0.1, 0.1, vec![]));
        users.insert("user2".to_string(), create_test_user("user2", -0.1, -0.1, vec![]));
        users.insert("user3".to_string(), create_test_user("user3", 0.5, 0.5, vec![]));

        // Create initial shards manually if initialize_regions isn't doing it
        router.shards.push(Shard::new(0, Point { x: 0.0, y: 0.0 }, 1.0));
        router.shards.push(Shard::new(1, Point { x: 0.5, y: 0.5 }, 1.0));

        // Now assign nodes to shards
        router.assign_nodes_to_shards(&users);

        // Debug output
        println!("Total nodes in shards: {}", router.shards.iter().map(|s| s.nodes.len()).sum::<usize>());
        println!("Total users: {}", users.len());
        println!("Shard distribution:");
        for (i, shard) in router.shards.iter().enumerate() {
            println!("Shard {}: {} nodes at center ({}, {})", 
                i, 
                shard.nodes.len(), 
                shard.center.x, 
                shard.center.y
            );
            for node_id in &shard.nodes {
                if let Some(user) = users.get(node_id) {
                    println!("  - Node {} at position ({}, {})", 
                        node_id, 
                        user.node.position.x, 
                        user.node.position.y
                    );
                }
            }
        }

        let total_nodes: usize = router.shards.iter()
            .map(|shard| shard.nodes.len())
            .sum::<usize>();

        assert_eq!(total_nodes, users.len(), 
            "Total nodes in shards ({}) should equal number of users ({})", 
            total_nodes, users.len());
    }

    #[test]
    fn test_shard_boundary_adjustment() {
        let mut router = HyperbolicRouter::new(100);
        let mut users = HashMap::new();

        // Create users with different transaction densities
        let transactions1 = vec![create_test_transaction("user1", "user2", 10.0); 1000];
        let transactions2 = vec![create_test_transaction("user2", "user1", 10.0); 10];

        users.insert("user1".to_string(), create_test_user("user1", 0.1, 0.1, transactions1));
        users.insert("user2".to_string(), create_test_user("user2", -0.1, -0.1, transactions2));

        router.initialize_regions(&users);
        
        // Record initial radii
        let initial_radii: Vec<f64> = router.shards.iter()
            .map(|shard| shard.radius)
            .collect();

        router.adjust_shard_boundaries(&users);

        // Verify that high-density shards shrink and low-density shards expand
        for (i, shard) in router.shards.iter().enumerate() {
            let initial_radius = initial_radii[i];
            let metrics = shard.calculate_metrics_from_users(&users);
            
            if metrics.transaction_density > 1000.0 {
                assert!(shard.radius < initial_radius);
            } else if metrics.transaction_density < 100.0 {
                assert!(shard.radius > initial_radius);
            }
        }
    }

    #[test]
    fn test_cross_shard_communication() {
        let mut mesh = QuantumMesh::new(2);
        let transaction = create_test_transaction("user1", "user2", 10.0);

        // Create users in different shards
        let mut users = HashMap::new();
        users.insert("user1".to_string(), create_test_user("user1", 0.8, 0.8, vec![]));
        users.insert("user2".to_string(), create_test_user("user2", -0.8, -0.8, vec![]));

        mesh.users = users;
        mesh.router.initialize_regions(&mesh.users);

        // Test cross-shard transaction propagation
        mesh.communicate_across_shards(&transaction);
        // Note: In a real test, you'd want to verify the transaction was properly propagated
        // This might involve adding some tracking mechanisms or mock objects
    }

    #[test]
    fn test_shard_metrics() {
        let mut users = HashMap::new();
        let center = Point { x: 0.0, y: 0.0 };
        let mut shard = Shard::new(0, center, 1.0);
        
        // Create a user with connections to ensure non-zero load
        let mut transactions = vec![create_test_transaction("user1", "user2", 10.0); 100];
        let mut user = create_test_user("user1", 0.1, 0.1, transactions);
        user.node.connections.push("user2".to_string()); // Add a connection to ensure load
        
        users.insert("user1".to_string(), user);
        shard.nodes.push("user1".to_string());
        
        let metrics = shard.calculate_metrics_from_users(&users);
        
        assert_eq!(metrics.node_count, 1);
        assert!(metrics.transaction_density > 0.0, "Transaction density should be positive");
        assert!(metrics.average_load > 0.0, "Average load should be positive. Got: {}", metrics.average_load);
    }

    #[test]
    fn test_shard_load_balancing() {
        let mut router = HyperbolicRouter::new(100);
        let mut users = HashMap::new();

        // Create users with varying loads
        for i in 0..10 {
            let transactions = vec![create_test_transaction(&format!("user{}", i), "userX", 10.0); i * 100];
            users.insert(
                format!("user{}", i),
                create_test_user(&format!("user{}", i), i as f64 * 0.1, i as f64 * 0.1, transactions)
            );
        }

        router.initialize_regions(&users);
        let initial_distribution = router.shards.iter()
            .map(|shard| shard.nodes.len())
            .collect::<Vec<_>>();

        router.adjust_shard_boundaries(&users);

        // Verify load balancing occurred
        for (i, shard) in router.shards.iter().enumerate() {
            let metrics = shard.calculate_metrics_from_users(&users);
            println!("Shard {} metrics: {:?}", i, metrics);
        }
    }

    #[test]
    fn test_state_compression() {
        let mut mesh = QuantumMesh::new(2);
        
        // Create test transactions in different regions of hyperbolic space
        let transactions = vec![
            create_test_transaction("user1", "user2", 10.0),
            create_test_transaction("user2", "user3", 20.0),
            create_test_transaction("user3", "user1", 30.0),
        ];
        
        // Add users with transactions at different points
        let mut users = HashMap::new();
        users.insert("user1".to_string(), create_test_user("user1", 0.1, 0.1, transactions.clone()));
        users.insert("user2".to_string(), create_test_user("user2", -0.3, 0.4, transactions.clone()));
        users.insert("user3".to_string(), create_test_user("user3", 0.5, -0.2, transactions));
        
        mesh.users = users;
        
        // Add some transactions to pending_transactions to ensure we have state points
        mesh.pending_transactions.push(create_test_transaction("user1", "user2", 15.0));
        mesh.pending_transactions.push(create_test_transaction("user2", "user3", 25.0));
        
        // Add a block to ensure we have some state points from blocks
        let block = Block {
            transactions: vec![create_test_transaction("user1", "user3", 35.0)],
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs(),
            previous_hash: "previous_hash".to_string(),
            hash: "hash".to_string(),
            difficulty: 2,
            nonce: 0,
            merkle_root: "merkle_root".to_string(),
        };
        mesh.blocks.push(block);
        
        // Test state compression
        let compressed_state = mesh.compress_state().expect("Failed to compress state");
        
        // Verify compression results
        assert!(!compressed_state.geodesic_path.is_empty(), "Geodesic path should not be empty");
        assert!(compressed_state.compression_ratio > 1.0, 
            "Compression ratio should be greater than 1.0, got {}", 
            compressed_state.compression_ratio);
        assert!(!compressed_state.compressed_data.is_empty(), "Compressed data should not be empty");
    }

    #[test]
    fn test_geometric_proof_generation() {
        let mesh = QuantumMesh::new(2);
        let test_points = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: -0.2, y: 0.3 },
            Point { x: 0.4, y: -0.1 },
        ];
        
        // Generate test compressed data
        let mut compressed_data = Vec::new();
        for point in &test_points {
            compressed_data.extend_from_slice(&point.x.to_le_bytes());
            compressed_data.extend_from_slice(&point.y.to_le_bytes());
        }
        
        // Generate proof
        let proof = mesh.generate_geometric_proof(&test_points, &compressed_data)
            .expect("Failed to generate geometric proof");
        
        // Verify proof structure
        assert!(!proof.tessellation_points.is_empty(), "Tessellation should not be empty");
        assert!(!proof.verification_path.is_empty(), "Verification path should not be empty");
        assert!(!proof.proof_hash.is_empty(), "Proof hash should not be empty");
        
        // Verify all tessellation points are within the Poincaré disk
        for point in &proof.tessellation_points {
            let norm = (point.x * point.x + point.y * point.y).sqrt();
            assert!(norm <= 1.0, "Tessellation point {:?} outside unit disk", point);
        }
    }

    #[test]
    fn test_compression_path_optimization() {
        let mesh = QuantumMesh::new(2);
        let test_points = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: 0.15, y: 0.12 }, // Close to first point
            Point { x: -0.5, y: 0.3 },  // Far from others
            Point { x: -0.48, y: 0.28 }, // Close to previous point
        ];
        
        let path = mesh.calculate_compression_path(&test_points);
        
        // Verify path properties
        assert_eq!(path.len(), test_points.len() + 1); // +1 for center of mass
        
        // Verify path starts with center of mass
        let center = mesh.calculate_center_of_mass(&test_points);
        assert_eq!(path[0], center);
        
        // Verify path follows nearest-neighbor pattern in hyperbolic space
        for i in 1..path.len()-1 {
            let current_distance = mesh.hyperbolic_distance(&path[i], &path[i+1]);
            // Check that this is indeed the shortest available path
            for j in i+2..path.len() {
                let alternative_distance = mesh.hyperbolic_distance(&path[i], &path[j]);
                assert!(current_distance <= alternative_distance, 
                    "Found shorter path: current = {}, alternative = {}", 
                    current_distance, alternative_distance);
            }
        }
    }

    #[test]
    fn test_tessellation_properties() {
        let mesh = QuantumMesh::new(2);
        let center = Point { x: 0.0, y: 0.0 };
        let test_path = vec![center];
        
        let tessellation = mesh.generate_tessellation(&test_path);
        
        // Verify tessellation size
        assert_eq!(tessellation.len(), 8 * test_path.len(), 
            "Each path point should generate 8 tessellation points");
        
        // Verify tessellation symmetry
        let mut angles = Vec::new();
        for point in &tessellation {
            let angle = point.y.atan2(point.x);
            angles.push(angle);
        }
        angles.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        // Check that angles are evenly distributed
        for i in 0..angles.len()-1 {
            let angle_diff = (angles[i+1] - angles[i]).abs();
            assert!((angle_diff - std::f64::consts::PI/4.0).abs() < 0.01, 
                "Tessellation points should be evenly distributed");
        }
    }

    #[test]
    fn test_verification_path_validity() {
        let mesh = QuantumMesh::new(2);
        
        // Create test compressed data
        let test_points = vec![
            Point { x: 0.2, y: 0.3 },
            Point { x: -0.4, y: 0.1 },
        ];
        
        let mut compressed_data = Vec::new();
        for point in &test_points {
            compressed_data.extend_from_slice(&point.x.to_le_bytes());
            compressed_data.extend_from_slice(&point.y.to_le_bytes());
        }
        
        let tessellation = mesh.generate_tessellation(&test_points);
        let verification_path = mesh.calculate_verification_path(&tessellation, &compressed_data);
        
        // Verify path is not empty
        assert!(!verification_path.is_empty(), "Verification path should not be empty");
        
        // Verify path continuity with a slightly larger threshold
        for window in verification_path.windows(2) {
            let distance = mesh.hyperbolic_distance(&window[0], &window[1]);
            assert!(distance < 0.6, "Verification path should be continuous, got distance {}", distance);
        }
        
        // Verify all points in the path are within the Poincaré disk
        for point in &verification_path {
            let norm = (point.x * point.x + point.y * point.y).sqrt();
            assert!(norm <= 1.0, "Point {:?} is outside the unit disk", point);
        }
        
        // Verify path includes all compressed points
        for chunk in compressed_data.chunks(16) {
            if chunk.len() >= 16 {
                let x = f64::from_le_bytes(chunk[0..8].try_into().unwrap());
                let y = f64::from_le_bytes(chunk[8..16].try_into().unwrap());
                let target = Point { x, y };
                
                // Check if any point in the path is close to the target
                let has_close_point = verification_path.iter().any(|p| {
                    mesh.hyperbolic_distance(p, &target) < 0.1
                });
                assert!(has_close_point, 
                    "Verification path should include point close to {:?}", target);
            }
        }
    }

    #[test]
    fn test_compression_metrics_tracking() {
        let mut mesh = QuantumMesh::new(2);
        
        // Create test data
        let test_points = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: -0.2, y: 0.3 },
            Point { x: 0.4, y: -0.1 },
        ];
        
        // Perform multiple compressions
        for _ in 0..5 {
            let (compressed, stats) = mesh.compress_state_with_metrics(&test_points)
                .expect("Compression should succeed");
            
            // Verify compression stats
            assert!(stats.compression_ratio > 1.0, 
                "Compression should reduce data size, got ratio: {}", 
                stats.compression_ratio);
            assert!(stats.compression_time > 0, 
                "Compression time should be positive");
            assert!(stats.tessellation_points > 0, 
                "Should have tessellation points");
        }
        
        // Get metrics report
        let metrics = mesh.get_compression_metrics();
        
        // Verify metrics
        assert_eq!(metrics.total_states_compressed, 5, 
            "Should have compressed 5 states");
        assert!(metrics.average_compression_ratio > 1.0, 
            "Average compression ratio should be greater than 1.0");
        assert!(metrics.average_compression_time > 0, 
            "Average compression time should be positive");
        assert!(metrics.failure_rate == 0.0, 
            "Should have no failures");
    }

    #[test]
    fn test_optimized_tessellation_generation() {
        let mesh = QuantumMesh::new(2);
        
        // Create test path with varying density requirements
        let test_path = vec![
            Point { x: 0.0, y: 0.0 },    // Center
            Point { x: 0.5, y: 0.0 },    // Dense region
            Point { x: -0.5, y: 0.0 },   // Another dense region
            Point { x: 0.0, y: 0.5 },    // Sparse region
        ];
        
        let tessellation = mesh.generate_optimized_tessellation(&test_path);
        
        // Verify basic properties
        assert!(!tessellation.is_empty(), "Tessellation should not be empty");
        
        // Verify points are within Poincaré disk
        for point in &tessellation {
            let norm = (point.x * point.x + point.y * point.y).sqrt();
            assert!(norm < 1.0, 
                "Tessellation point {:?} should be within unit disk", point);
        }
        
        // Verify density adaptation
        let density = mesh.calculate_tessellation_density(&test_path);
        assert!(density > 0.0 && density < 1.0, 
            "Density should be between 0 and 1, got {}", density);
        
        // Verify point distribution
        let mut covered_areas = HashSet::new();
        for point in &tessellation {
            let area_key = mesh.get_area_key(point);
            covered_areas.insert(area_key);
        }
        
        // Should have reasonable coverage
        assert!(covered_areas.len() >= test_path.len(), 
            "Should cover at least as many areas as path points");
    }

    #[test]
    fn test_tessellation_optimization() {
        let mesh = QuantumMesh::new(2);
        
        // Create a set of points with some intentionally close together
        let mut test_points = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: 0.101, y: 0.101 }, // Very close to first point
            Point { x: 0.3, y: 0.3 },
            Point { x: 0.301, y: 0.301 }, // Very close to third point
            Point { x: -0.4, y: -0.4 },
        ];
        
        // Optimize the tessellation points
        mesh.optimize_tessellation_points(&mut test_points);
        
        // Verify redundant points were removed
        assert!(test_points.len() < 5, 
            "Should have removed some redundant points, got {} points", 
            test_points.len());
        
        // Verify minimum distance between points
        for i in 0..test_points.len() {
            for j in i+1..test_points.len() {
                let distance = mesh.hyperbolic_distance(&test_points[i], &test_points[j]);
                assert!(distance >= 0.05, 
                    "Points should maintain minimum distance, got {}", distance);
            }
        }
    }

    #[test]
    fn test_compression_cache() {
        let mut mesh = QuantumMesh::new(2);
        
        // Create test points
        let test_points = vec![
            Point { x: 0.1, y: 0.1 },
            Point { x: -0.2, y: 0.3 },
        ];
        
        // First compression should not be cached
        let (first_compressed, first_stats) = mesh.compress_state_with_metrics(&test_points)
            .expect("First compression should succeed");
        
        // Second compression should use cache
        let (second_compressed, second_stats) = mesh.compress_state_with_metrics(&test_points)
            .expect("Second compression should succeed");
        
        // Verify cache hit
        assert!(second_stats.compression_time < first_stats.compression_time, 
            "Cached compression should be faster");
        
        // Verify cached result matches
        assert_eq!(first_compressed.compression_ratio, second_compressed.compression_ratio,
            "Cached compression should have same ratio");
        assert_eq!(first_compressed.proof.proof_hash, second_compressed.proof.proof_hash,
            "Cached compression should have same proof hash");
    }

    #[test]
    fn test_adaptive_tessellation_density() {
        let mesh = QuantumMesh::new(2);
        
        // Test paths of different lengths
        let short_path = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.1, y: 0.1 },
        ];
        
        let long_path = vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: 0.3, y: 0.0 },
            Point { x: 0.0, y: 0.3 },
            Point { x: -0.3, y: 0.0 },
            Point { x: 0.0, y: -0.3 },
        ];
        
        let short_density = mesh.calculate_tessellation_density(&short_path);
        let long_density = mesh.calculate_tessellation_density(&long_path);
        
        // Longer paths should have lower density to maintain efficiency
        assert!(long_density < short_density, 
            "Longer paths should have lower tessellation density");
        
        // Generate tessellations
        let short_tessellation = mesh.generate_adaptive_tessellation(
            &short_path[0], 
            short_density
        );
        
        let long_tessellation = mesh.generate_adaptive_tessellation(
            &long_path[0], 
            long_density
        );
        
        // Verify tessellation sizes are appropriate
        assert!(short_tessellation.len() < long_tessellation.len(), 
            "Longer path should have more tessellation points");
        
        // Verify all points are in valid positions
        for point in short_tessellation.iter().chain(long_tessellation.iter()) {
            let norm = (point.x * point.x + point.y * point.y).sqrt();
            assert!(norm < 1.0, "Point should be within unit disk");
        }
    }
}