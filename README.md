# DIFAI: Differentiable Active Inference Framework

A JAX-based implementation of Active Inference for continuous control and state estimation, enabling efficient Bayesian learning and decision-making in uncertain environments.

## Overview

DIFAI provides a complete framework for implementing Active Inference agents that can:
- Learn and maintain probabilistic beliefs about system states, dynamics, and observations
- Select actions to minimize expected free energy through planning
- Handle various noise sources including observation noise and system uncertainty
- Support both inference-only and active control scenarios

The framework is built on JAX for high-performance computation with automatic differentiation and JIT compilation.

## Key Features

- **Bayesian State Estimation**: Unscented Kalman Filter (UKF) and variational inference for belief updates
- **Active Control**: Expected Free Energy minimization for action selection
- **Flexible Environment Interface**: Easy-to-extend base classes for custom environments
- **High Performance**: JAX-based implementation with JIT compilation
- **Configurable**: YAML-based configuration system with sensible defaults

## Installation

We recommend using a fresh virtual environment with Python>=3.11. For the best performance, we recommend using a CUDA supported GPU and installing difai with the "\[gpu\]" add-on.

```bash
git clone https://github.com/mkl4r/difai-base.git
cd difai-base
```

### With GPU Support (recommended)

```bash
pip install -e ".[gpu]"
```

### With TPU Support

```bash
pip install -e ".[tpu]"
```

### CPU Only

```bash
pip install -e ".[cpu]"
```

## Quick Start

### Complete Minimal Example

See `difai/example/minimal_example.py` for a complete working example featuring:
- Simple 1D spring-mass system
- Both inference-only and active control modes
- Noise handling and belief updating
- Visualization of results

### Running the Example

```bash
cd difai/example
python minimal_example.py
```

This will generate plots showing:
- State estimation accuracy
- Belief uncertainty evolution
- Action selection for target reaching


## Core Components

### 1. AIF_Env (Environment Interface)

Base class for defining the generative process/model. To create a new environment, implement the system dynamics in `_forward_complete`  and observations in `_get_observation_complete`. Do not forget to set the corresponding system parameters and dimensions in `__init__`.

```python
class MyEnv(AIF_Env):
    @staticmethod
    def _forward_complete(x, u, dt, random_realisation, key, **sys_params):
        """Define system dynamics: x_{t+1} = f(x_t, u_t, θ)"""
        pass
    
    @staticmethod
    def _get_observation_complete(x, **sys_params):
        """Define observation model: o_t = g(x_t, θ)"""
        pass
```

### 2. AIF_Agent (Active Inference Agent)

Implements Bayesian inference and action selection:

- **Belief Updates**: Maintains probabilistic beliefs over states, system parameters, and noise
- **Action Selection**: Minimizes expected free energy through planning
- **Configuration**: Configurable via parameters and YAML files

### 3. AIF_Simulation (Simulation Engine)

Run agent-environment interactions:

- **Multiple Run Modes**: Inference-only, active control, perceptual delays
- **Noise Handling**: Supports various noise sources and configurations
- **Logging**: Comprehensive data collection for analysis

## Configuration

### Parameter Configuration

The framework uses a hierarchical configuration system:

1. **Default parameters** (loaded from `default_config.yaml`)
2. **User parameters** (can override defaults)
3. **Runtime parameters** (set programmatically)

Key parameter categories:

```python
# Action Planning & Control
a_lims=None                    # Action limits [lower_bounds, upper_bounds]
n_plans=2000                   # Number of action sequences to evaluate during planning
horizon=20                     # Planning horizon (number of future steps)
multistep=1                    # Control update frequency (apply same action for N steps)
select_max_pi=True             # Select best plan (True) vs sample from plan distribution (False)

# Belief Updating - Observations
n_samples_o=300                # Number of samples for observation-based belief updates
n_steps_o=100                  # Optimization steps for variational inference after observations
lr_o=0.0001                    # Learning rate for observation updates
standardise_state=False        # Standardize states using scaling factors
state_scaling=None             # Scaling factors for state dimensions

# Belief Updating - Actions & Dynamics
use_complete_ukf=True          # Use Unscented Kalman Filter for belief updates
exp_normal_sys_params=False    # Whether system parameters are log-normal distributed
n_samples_a=30                 # Number of state samples for action-based belief updates (only if use_complete_ukf=False)
n_samples_a_combine=200        # Samples for combining beliefs (only if use_complete_ukf=False)
n_samples_a_noise_sys=100      # Samples for noise/system parameter estimation (only if use_complete_ukf=False)

# Active Inference 
use_info_gain=False            # Include information gain in action selection
use_observation_preference=True # Use observation-based preferences (vs state-based)
use_pragmatic_value=True       # Include pragmatic value in action selection
scale_pragmatic_value=1        # Scaling factor for pragmatic value vs info gain
use_fixed_plans=False          # Use fixed set of plans vs random sampling
action_prior=None              # Prior distribution over actions [mean, cov]

# Information Gain Settings
n_samples_ig_s=3               # State samples for information gain calculation
n_samples_ig_o=3               # Observation samples for information gain calculation

# Observation Preference Settings
n_samples_obs_pref_s=100       # State samples for observation preference calculation
n_samples_obs_pref_o=10        # Observation samples per state for preference calculation
C_index=None                   # Index of observation dimension for preference (must be set)
sys_dependent_C=None           # Make preference depend on system parameters [C indices, sys indices]
state_dependent_C=None         # Make preference depend on state [C indices, state indices]

# Advanced Features
reaction_time=0.0              # Sensorimotor delay in seconds

```

## Noise Modeling

DIFAI supports multiple noise sources:

### Observation Noise
```python
noise_params = {
    'observation_std': {
        'id': jnp.array([0]),        # Which observations are affected
        'value': jnp.array([0.05])   # Noise standard deviation
    }
}
```

### Motor Noise
```python
noise_params = {
    'constant_motor_noise': {
        'id': jnp.array([0]),        # Which actions are affected
        'value': jnp.array([0.1])    # Noise standard deviation
    },
    'signal_dependent_noise': {
        'id': jnp.array([0]),        # Noise parameter index
        'value': jnp.array([0.02])   # Proportional noise factor
    }
}
```

## Advanced Features

### Perceptual Delays
Model sensorimotor delays:

```python
agent.set_params(reaction_time=0.1)  # 100ms delay
results = sim.run_aif_perceptual_delay(numsteps=100)
```

### Custom Action Priors
Incorporate preference about actions:

```python
action_prior = [jnp.zeros(1), jnp.eye(1) * 0.1]  # Mean and covariance
agent.set_params(action_prior=action_prior)
```

## Performance Tips

1. **JIT Compilation**: The framework automatically JITs critical functions
2. **Batch Sizes**: Adjust `n_samples_*` parameters based on your hardware
3. **Planning Horizon**: Balance accuracy vs. computational cost with `horizon` and `n_plans`
4. **GPU Usage**: Install with `[gpu]` for CUDA acceleration

## API Reference

### Core Classes

- **`AIF_Env`**: Base environment class
- **`AIF_Agent`**: Active inference agent implementation  
- **`AIF_Simulation`**: Simulation orchestration

### Key Methods

- **`agent.set_params(**kwargs)`**: Configure agent parameters
- **`agent.set_initial_beliefs(state, sys, noise)`**: Set prior beliefs
- **`agent.set_preference_distribution(C, C_index)`**: Define goals
- **`sim.run_aif_perceptual_delay(numsteps)`**: Run active control simulation
- **`sim.run_inference_only(numsteps)`**: Run state estimation only


## License

This project is licensed under the terms specified in the LICENSE file.

## Dependencies

- **JAX**: Numerical computing and automatic differentiation
- **NumPy**: Numerical arrays and operations  
- **Optax**: Gradient-based optimization
- **PyYAML**: Configuration file parsing
- **tqdm**: Progress bars
- **Matplotlib**: Plotting and visualization
