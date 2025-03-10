use std::hash::{Hash, Hasher};
use serde::{Serialize, Deserialize};

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Point {
    pub x: f64,
    pub y: f64,
}

impl PartialEq for Point {
    fn eq(&self, other: &Self) -> bool {
        self.x.to_bits() == other.x.to_bits() && 
        self.y.to_bits() == other.y.to_bits()
    }
}

impl Eq for Point {}

impl Hash for Point {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.x.to_bits().hash(state);
        self.y.to_bits().hash(state);
    }
}

// Define the traits that main.rs is trying to implement
pub trait HasCoordinates {
    fn get_x(&self) -> f64;
    fn get_y(&self) -> f64;
}

// Implement the crypto traits for our domain types
impl HasCoordinates for Point {
    fn get_x(&self) -> f64 { self.x }
    fn get_y(&self) -> f64 { self.y }
}

// Add hyperbolic distance calculation to the geometry module
pub fn hyperbolic_distance(p1: &Point, p2: &Point) -> f64 {
    let dx = p2.x - p1.x;
    let dy = p2.y - p1.y;
    let euclidean_dist = (dx * dx + dy * dy).sqrt();
    
    // Add safeguard for numerical stability
    if euclidean_dist >= 0.99 {
        return f64::MAX;  // Or some large finite value
    }
    
    2.0 * ((1.0 + euclidean_dist) / (1.0 - euclidean_dist)).ln()
}