# Coherence Gate Scaling

Code and data for:

**"Thermodynamic Scaling Limits of Algorithmic Oscillatory Synchronization: Why Physical Coupling is Required for Large-Scale Oscillatory Neural Networks"**

Ian Todd, Sydney Medical School, University of Sydney

## Abstract

Digital simulation of oscillatory neural networks (e.g., HoloGraph) incurs O(N²) computational cost per timestep. Physical instantiation of coupled oscillators eliminates this barrier, achieving 10²–10³× power reduction with advantages scaling as O(N²/K).

## Contents

- `power_comparison_v2.py` — Simulation comparing digital vs physical coupling energy
- `figures/` — Publication figures

## Usage

```bash
python power_comparison_v2.py
```

Generates:
- Energy vs time comparison (N=100, 86× advantage)
- N-scaling plot (24× to 224× advantage)

## Key Results

| N | Digital (J) | Physical (J) | Advantage |
|---|-------------|--------------|-----------|
| 25 | 1.07×10⁻⁵ | 4.40×10⁻⁷ | 24× |
| 50 | 2.31×10⁻⁵ | 4.60×10⁻⁷ | 50× |
| 100 | 4.30×10⁻⁵ | 5.00×10⁻⁷ | 86× |
| 200 | 8.22×10⁻⁵ | 5.80×10⁻⁷ | 142× |
| 400 | 1.66×10⁻⁴ | 7.40×10⁻⁷ | 224× |

## License

MIT
