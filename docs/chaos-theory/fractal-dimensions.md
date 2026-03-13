# Fractal Dimensions

## The Problem

You are a geographer measuring the length of a coastline. You start with a 100 km ruler
and measure 500 km. Then you switch to a 10 km ruler, following every bay and inlet, and
measure 700 km. A 1 km ruler gives 950 km. The measured length keeps growing as your
ruler shrinks -- it never converges. This is Richardson's paradox, and the reason is that
coastlines are fractals: their complexity exists at every scale.

You need a single number that captures "how rough" or "how space-filling" a shape is.
That number is the fractal dimension. It appears in texture analysis (satellite imagery),
material science (surface roughness), financial volatility analysis, and attractor
characterisation in dynamical systems.

## The Intuition

A straight line has dimension 1. A filled square has dimension 2. A coastline is
*between* 1 and 2: it is more than a line (it wiggles and fills more space) but less than
a filled area. The fractal dimension D tells you exactly how much "in between" it is.

- D ~ 1.0: Nearly a smooth line.
- D ~ 1.25: A mildly jagged coastline (Britain ~ 1.25).
- D ~ 1.5: Very jagged, like a random walk.
- D ~ 2.0: So convoluted it nearly fills the plane.

The classic strange attractors have non-integer dimensions too: the Lorenz attractor has
D ~ 2.06 (slightly more than a sheet, due to its fractal layering).

## How It Works

### Box-counting dimension

```
D_box = lim_{eps->0} log(N(eps)) / log(1/eps)
```

**In plain English:** Cover the set with boxes of side length eps. Count how many boxes
N(eps) contain at least one point. Shrink the boxes and repeat. Plot log(N) vs
log(1/eps); the slope of the best-fit line is the fractal dimension.

### Correlation dimension

```
C(r) = (2 / N(N-1)) * #{pairs where |x_i - x_j| < r}
D_corr = lim_{r->0} log(C(r)) / log(r)
```

**In plain English:** For each radius r, count what fraction of all point pairs are
closer than r. As r shrinks, this fraction decreases as a power law. The exponent of
that power law is the correlation dimension.

The correlation dimension is related to but slightly different from the box-counting
dimension. For most practical attractors: D_corr <= D_box.

### Hurst exponent (related)

```
H = slope of log(R/S) vs log(n)
```

**In plain English:** For a time series, the Hurst exponent measures long-range
dependence. H = 0.5 means random walk; H > 0.5 means trending (persistent); H < 0.5
means mean-reverting (anti-persistent). The fractal dimension of the time series graph
is D = 2 - H.

## In Rust

```rust
use machin_chaos::fractal::{
    box_counting_dimension_2d,
    correlation_dimension,
    hurst_exponent,
};

// --- Box-counting: dimension of a line ---
let line: Vec<(f64, f64)> = (0..1000)
    .map(|i| {
        let t = i as f64 / 999.0;
        (t, t)  // diagonal line
    })
    .collect();
let dim = box_counting_dimension_2d(&line, 8);
println!("Line dimension: {:.2}", dim);  // ~1.0

// --- Box-counting: dimension of a filled square ---
let mut square = Vec::new();
for i in 0..50 {
    for j in 0..50 {
        square.push((i as f64 / 49.0, j as f64 / 49.0));
    }
}
let dim = box_counting_dimension_2d(&square, 6);
println!("Square dimension: {:.2}", dim);  // ~2.0

// --- Box-counting: Henon attractor ---
use machin_chaos::attractors::{henon, HenonParams};
let traj = henon(0.1, 0.1, &HenonParams::default(), 50_000);
let points: Vec<(f64, f64)> = traj[1000..].to_vec(); // discard transient
let dim = box_counting_dimension_2d(&points, 10);
println!("Henon attractor dimension: {:.2}", dim);  // ~1.26

// --- Correlation dimension ---
// Convert 2D points to vector-of-vectors format
let data: Vec<Vec<f64>> = points.iter()
    .take(2000)  // subsample for speed
    .map(|&(x, y)| vec![x, y])
    .collect();
let d_corr = correlation_dimension(&data, 0.001, 1.0, 20);
println!("Correlation dimension: {:.2}", d_corr);

// --- Hurst exponent of a time series ---
let prices: Vec<f64> = (0..1024)
    .scan(100.0, |state, _| {
        *state += 0.1 * (*state * 0.01);  // trending series
        Some(*state)
    })
    .collect();
let h = hurst_exponent(&prices);
println!("Hurst exponent: {:.2}", h);  // > 0.5 for trending data
```

## When To Use This

| Technique | Best for | Limitations |
|-----------|----------|-------------|
| **Box-counting** | 2D point sets, attractor shapes, image textures | Sensitive to the number of points and scale range |
| **Correlation dimension** | High-dimensional attractors, embedding dimension estimation | O(N^2) computation; needs many data points |
| **Hurst exponent** | Time series: trend detection, volatility analysis | Assumes self-similarity; finite-sample bias |

## Key Parameters

| Parameter | What it controls | Rule of thumb |
|-----------|-----------------|---------------|
| `num_scales` | Number of box sizes / radius values to test | 6--12 for box counting; 15--25 for correlation dimension |
| `r_min, r_max` | Range of radii for correlation dimension | r_min ~ smallest inter-point distance; r_max ~ dataset extent |
| Number of points | Statistical reliability | Box counting: 1000+; correlation dimension: 2000+; Hurst: 256+ |

## Pitfalls

1. **Too few points.** Fractal dimension estimates are biased with small samples. For
   box counting, the set should densely fill the fractal structure. Thousands of points
   are the minimum for reliable estimates.

2. **Scale range selection.** The dimension is estimated from the *slope* of a log-log
   plot. If you include scales that are too large (finite-size effects) or too small
   (discretisation effects), the slope will be wrong. Inspect the log-log plot visually
   if possible.

3. **Lacunarity.** Two fractals can have the same dimension but very different visual
   appearance (gappiness). The fractal dimension alone does not fully characterise a
   fractal.

4. **Hurst exponent bias.** The R/S method has known bias for short time series.
   Series shorter than ~256 points will produce unreliable H estimates.

5. **Correlation dimension O(N^2).** Computing all pairwise distances is expensive.
   For large datasets, subsample to a few thousand points or use approximate methods.

## Going Further

- Compute the dimension of Lorenz attractor trajectories by projecting 3D points to 2D
  (e.g., x-z plane) and calling `box_counting_dimension_2d`.
- Use `machin_chaos::lyapunov::lyapunov_spectrum` alongside fractal dimension to validate
  the Kaplan-Yorke conjecture: D_KY = j + sum(lambda_1..lambda_j) / |lambda_{j+1}|.
- For time series, use `machin_chaos::embedding` to reconstruct a phase-space attractor
  via delay embedding, then measure its correlation dimension.
- Compare the Hurst exponent of financial returns across different time windows to detect
  regime changes.
