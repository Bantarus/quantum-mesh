use crate::quantum_crypto::Point;

pub trait HyperbolicDistance {
    fn hyperbolic_distance(p1: &Point, p2: &Point) -> f64 {
        let dx = p2.x - p1.x;
        let dy = p2.y - p1.y;
        let euclidean_dist = (dx * dx + dy * dy).sqrt();

        // Add safeguard for numerical stability
        if euclidean_dist >= 0.99 {
            return f64::MAX;  // Or some large finite value
        }
        
        2.0 * ((1.0 + euclidean_dist) / (1.0 - euclidean_dist)).ln()
    }
}