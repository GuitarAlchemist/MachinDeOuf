# Use Case: GIS & Spatial Analysis

> Combining Kalman filters, DBSCAN, A*, FFT, and HMMs for GPS tracking, spatial clustering, route optimization, and terrain analysis.

## The Scenario

You're building a logistics platform that manages a fleet of 500 delivery vehicles. You need to:

1. **Smooth GPS tracks** — raw GPS jumps around; you need clean trajectories (Kalman filter)
2. **Find delivery hotspots** — cluster stop locations to identify warehouses and frequent destinations (DBSCAN)
3. **Optimize routes** — find shortest paths through the road network (A*)
4. **Analyze terrain** — detect periodic patterns in elevation data for road quality assessment (FFT)
5. **Snap GPS to roads** — match noisy GPS points to the most likely road segments (HMM/Viterbi)
6. **Detect anomalies** — identify unusual vehicle behavior patterns (Lyapunov exponents + Bloom filters)

## Step 1: GPS Track Smoothing with Kalman Filter

Raw GPS readings scatter ±10 meters due to atmospheric interference, multipath reflection, and sensor noise. The Kalman filter fuses noisy position readings with a motion model to produce smooth, accurate tracks.

```rust
use ix_signal::kalman::KalmanFilter;
use ndarray::array;

// Constant-velocity model: state = [x, vx, y, vy]
// GPS measures position only: observation = [x, y]
let dt = 1.0; // 1-second GPS updates
let mut kf = KalmanFilter::new(4, 2); // 4 state dims, 2 observation dims

// State transition: x_new = x + vx*dt, vx_new = vx (constant velocity)
kf.transition = array![
    [1.0, dt,  0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, dt ],
    [0.0, 0.0, 0.0, 1.0],
];

// Observation: we see x and y, not velocities
kf.observation = array![
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
];

// Noise: GPS has ~10m accuracy, vehicle acceleration varies
kf.measurement_noise = array![[100.0, 0.0], [0.0, 100.0]]; // 10m std dev
kf.process_noise = array![
    [0.25, 0.5, 0.0, 0.0],
    [0.5,  1.0, 0.0, 0.0],
    [0.0,  0.0, 0.25, 0.5],
    [0.0,  0.0, 0.5,  1.0],
];

// Process noisy GPS readings
let gps_readings = vec![
    array![40.7128, -74.0060],  // NYC coordinates (simplified)
    array![40.7131, -74.0055],
    array![40.7150, -74.0040],  // Jump — probably noise
    array![40.7135, -74.0048],
    // ... hundreds more readings
];

let smoothed: Vec<_> = gps_readings.iter()
    .map(|reading| kf.step(reading, None))
    .collect();

// smoothed[i] = [x, vx, y, vy] — position AND estimated velocity
for (i, state) in smoothed.iter().enumerate() {
    println!("t={}: pos=({:.4}, {:.4}), speed=({:.2}, {:.2})",
        i, state[0], state[2], state[1], state[3]);
}
```

> See [Signal: Kalman Filter](../signal-processing/kalman-filter.md) for the full Kalman filter doc.

## Step 2: Spatial Clustering with DBSCAN

Find delivery hotspots — areas where vehicles frequently stop. DBSCAN is perfect because hotspots have irregular shapes (they follow buildings, loading docks, intersections) and you need to identify noise (one-off stops).

```rust
use ix_unsupervised::{DBSCAN, Clusterer};
use ndarray::array;

// Stop locations from fleet: [latitude, longitude]
let stops = array![
    // Warehouse cluster (Brooklyn)
    [40.6782, -73.9442], [40.6785, -73.9440], [40.6780, -73.9445],
    [40.6783, -73.9441], [40.6781, -73.9443],
    // Office district cluster (Midtown)
    [40.7549, -73.9840], [40.7551, -73.9838], [40.7548, -73.9842],
    [40.7550, -73.9839],
    // Noise: random one-off stops
    [40.7000, -73.9500], [40.8000, -73.9000],
];

// eps ≈ 0.001 degrees ≈ 100m at NYC latitude
let mut dbscan = DBSCAN::new(0.001, 3); // min 3 stops to form a cluster
let labels = dbscan.fit_predict(&stops);

// Label 0 = noise, 1+ = cluster ID
let n_clusters = *labels.iter().max().unwrap_or(&0);
let n_noise = labels.iter().filter(|&&l| l == 0).count();
println!("{} hotspots found, {} noise points", n_clusters, n_noise);

// Compute cluster centroids for each hotspot
for cluster_id in 1..=n_clusters {
    let points: Vec<_> = stops.outer_iter()
        .zip(labels.iter())
        .filter(|(_, &l)| l == cluster_id)
        .map(|(row, _)| row.to_owned())
        .collect();
    let centroid_lat = points.iter().map(|p| p[0]).sum::<f64>() / points.len() as f64;
    let centroid_lon = points.iter().map(|p| p[1]).sum::<f64>() / points.len() as f64;
    println!("Hotspot {}: ({:.4}, {:.4}) — {} stops",
        cluster_id, centroid_lat, centroid_lon, points.len());
}
```

> See [Unsupervised: DBSCAN](../unsupervised-learning/dbscan.md) for the full DBSCAN doc.

## Step 3: Route Optimization with A*

Find the shortest path between delivery stops on a road network. A* uses a heuristic (straight-line distance) to focus the search toward the goal.

```rust
use ix_search::astar::{SearchState, astar, SearchResult};

#[derive(Clone, Hash, Eq, PartialEq)]
struct RoadNode {
    id: usize,
    lat: i64,  // Fixed-point (lat * 10000) for Hash/Eq
    lon: i64,
}

impl SearchState for RoadNode {
    type Action = usize; // Edge ID

    fn successors(&self) -> Vec<(usize, Self, f64)> {
        // Return connected roads with travel time as cost
        get_road_neighbors(self.id)
            .iter()
            .map(|(edge_id, neighbor, distance_km)| {
                let travel_time = distance_km / 50.0; // Assume 50 km/h average
                (*edge_id, neighbor.clone(), travel_time)
            })
            .collect()
    }

    fn is_goal(&self) -> bool {
        self.id == DESTINATION_ID
    }
}

// Heuristic: straight-line distance / max_speed
let heuristic = |node: &RoadNode| -> f64 {
    let dx = (node.lat - dest_lat) as f64 / 10000.0;
    let dy = (node.lon - dest_lon) as f64 / 10000.0;
    let straight_line_km = (dx * dx + dy * dy).sqrt() * 111.0; // ~111km per degree
    straight_line_km / 80.0 // Optimistic: max 80 km/h
};

let start = RoadNode { id: 0, lat: 407128, lon: -740060 };
if let Some(result) = astar(&start, &heuristic) {
    println!("Route found: {} segments, {:.1} min", result.path.len(), result.cost * 60.0);
    println!("Nodes explored: {}", result.nodes_expanded);
}
```

> See [Search: A*](../search-and-graphs/astar-search.md) for the full A* doc.

## Step 4: Terrain Analysis with FFT

Analyze road elevation profiles to detect periodic patterns — potholes at regular intervals, speed bumps, or road surface quality.

```rust
use ix_signal::fft::{rfft, magnitude_spectrum, frequency_bins};

// Elevation readings every 1 meter along a road (sampled via lidar/GPS)
let elevation: Vec<f64> = load_elevation_profile("route_42.csv");
let sample_rate = 1.0; // 1 sample per meter

let spectrum = rfft(&elevation);
let magnitudes = magnitude_spectrum(&spectrum);
let freqs = frequency_bins(spectrum.len() * 2, sample_rate);

// Find dominant spatial frequencies
let mut peaks: Vec<(f64, f64)> = freqs.iter()
    .zip(magnitudes.iter())
    .filter(|(&f, _)| f > 0.001) // Skip DC component
    .map(|(&f, &m)| (f, m))
    .collect();
peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

println!("Top spatial frequencies:");
for (freq, mag) in peaks.iter().take(5) {
    let wavelength = 1.0 / freq;
    println!("  Wavelength: {:.1}m, Magnitude: {:.2}", wavelength, mag);
}
// Wavelength ~5m → speed bumps; ~0.5m → rough surface; ~50m → gentle hills
```

> See [Signal: FFT](../signal-processing/fft-intuition.md) for the full FFT doc.

## Step 5: Map Matching with HMM/Viterbi

Snap noisy GPS points to the most likely road segments. Hidden states = road segments, observations = GPS zones, transitions = road connectivity.

```rust
use ix_graph::hmm::HiddenMarkovModel;
use ndarray::{Array1, Array2};

// Build HMM from road network
// States: road segments, Observations: discretized GPS regions
let n_segments = 20;
let n_gps_zones = 10;

let initial = Array1::from_vec(vec![1.0 / n_segments as f64; n_segments]);
let transition = build_road_transition_matrix(n_segments); // From road connectivity
let emission = build_gps_emission_matrix(n_segments, n_gps_zones); // GPS accuracy model

let hmm = HiddenMarkovModel::new(initial, transition, emission).unwrap();

// Noisy GPS readings → discretized zone IDs
let gps_zones = vec![3, 3, 4, 5, 5, 6, 7, 7, 8, 8];

let (road_segments, log_prob) = hmm.viterbi(&gps_zones);
println!("GPS zones:     {:?}", gps_zones);
println!("Road segments: {:?}", road_segments);
println!("Confidence: {:.2}", log_prob);
```

> See [Sequence: Viterbi](../sequence-models/viterbi-algorithm.md) for the full Viterbi doc.

## Step 6: Anomaly Detection with Bloom Filters

Track which route patterns are "normal" using a Bloom filter. When a vehicle's route hash isn't in the filter, flag it for review.

```rust
use ix_probabilistic::BloomFilter;

// Train: insert all normal route patterns
let mut normal_routes = BloomFilter::new(10_000, 0.01); // 1% false positive rate

for route in historical_normal_routes {
    let route_hash = format!("{:?}", route.segment_sequence);
    normal_routes.insert(&route_hash);
}

// Monitor: check if current route is in the normal set
let current_route_hash = format!("{:?}", current_vehicle.segment_sequence);
if !normal_routes.contains(&current_route_hash) {
    println!("ALERT: Vehicle {} on unusual route!", current_vehicle.id);
}
```

> See [Probabilistic: Bloom Filters](../probabilistic-structures/bloom-filters.md) for the full doc.

## Algorithms Used

| Algorithm | Doc | Role in GIS |
|-----------|-----|-------------|
| Kalman Filter | [Signal: Kalman](../signal-processing/kalman-filter.md) | GPS track smoothing |
| DBSCAN | [Unsupervised: DBSCAN](../unsupervised-learning/dbscan.md) | Spatial clustering / hotspots |
| A* Search | [Search: A*](../search-and-graphs/astar-search.md) | Route optimization |
| FFT | [Signal: FFT](../signal-processing/fft-intuition.md) | Terrain frequency analysis |
| HMM/Viterbi | [Sequence: Viterbi](../sequence-models/viterbi-algorithm.md) | GPS-to-road map matching |
| Bloom Filter | [Probabilistic: Bloom](../probabilistic-structures/bloom-filters.md) | Route anomaly detection |
| K-Means | [Unsupervised: K-Means](../unsupervised-learning/kmeans.md) | Zone partitioning |
| Markov Chains | [Sequence: Markov](../sequence-models/markov-chains.md) | Traffic flow modeling |
| Lyapunov Exponents | [Chaos: Lyapunov](../chaos-theory/lyapunov-exponents.md) | Traffic chaos detection |
