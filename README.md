# Epistemic-Risk-Seeking Actor-Critic (ERSAC)

This repository provides an implementation of the **Epistemic-Risk-Seeking Actor-Critic (ERSAC)** algorithm, as introduced in [ERSAC](https://arxiv.org/abs/2302.09339), applied to the **DeepSea** environment [bsuite](https://arxiv.org/abs/1908.03568).

A full list of package versions used in our Conda environment is provided to ensure reproducibility.

---

## Example Usage

To train ERSAC with **uncertainty estimation based on explicit (state, action) visit counts** (i.e., without reward ensemble), you can run:

```python
training_data = run_ersac(
    noise_probability=0,
    seed=SEED,
    depth=i,
    learning_rate=0.001,
    gradient_clipping=5,
    init_tau=0.1,
    min_tau=0.0001,
    pow_seen_pairs=2,
    Lambda=0.8,
    uncertainty_scale=1,
    n_episode=100000,
    mid_size=200,
    reward_estimation=False,
    n_heads=10,
    saving_path=None
)

                                          
