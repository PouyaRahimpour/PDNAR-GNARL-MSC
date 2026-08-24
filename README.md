# Primal-Dual Neural Algorithmic Reasoning


### Installation

1. Create the environment using the `environment.yml` file:

    ```bash
    conda env create -f environment.yml
    ```

2. Activate the environment:

    ```bash
    conda activate pdnar
    ```

### Usage 

To run the experiments for the three NP-hard problems, use the following commands:

1. **Minimum Vertex Cover (MVC)**

    ```bash
    python main.py data.algorithm="vertex_cover"
    ```

2. **Minimum Set Cover (MSC)**

    ```bash
    python main.py data.algorithm="set_cover"
    ```

3. **Minimum Hitting Set (MHS)** 

    ```bash
    python main.py data.algorithm="hitting_set" model.model.eps=True 
    ```

- Note that  `model.model.eps` controls whether the uniform increase rule in used.

### Optional parameters 
| **Parameter**         | **Description**                                      | **Type**      | **Default Value** | 
|-----------------------|------------------------------------------------------|---------------|-------------------|
| `seed`        | Random seed.               | `int`         | `0`        | 
| `wandb_use`        | Whether to use wandb.                        | `bool`         | `False`        | 
| `inference_only`    | Whether to only perform inference.                          | `bool`         | `False`              | 
| `checkpoint`        | Path of pretrained model for inference only.                            | `str`         | `null`             |                      

### GNARL-MSC controlled baseline

The GNARL implementation lives in `src/gnarl/` and intentionally leaves
`src/model/` untouched. It consumes the same serialized MSC records as PDNAR:
the set weights, element--set bipartite incidence, exact ILP optima, and the
primal-dual solution used as an evaluation baseline. Its only initial change is
the solver: sequential set selection in an MDP replaces PDNAR's NAR execution.

```bash
python scripts/train_bc.py --output runs/gnarl_msc_bc.pt
python scripts/train_ppo.py --checkpoint runs/gnarl_msc_bc.pt --output runs/gnarl_msc_ppo.pt
python scripts/evaluate.py runs/gnarl_msc_ppo.pt
```

`train_bc.py` clones a uniform action distribution over still-unselected sets
in PDNAR's stored ILP optimum. `train_ppo.py` can run from scratch or fine-tune
that policy using only incremental negative set cost. Evaluation reports
feasibility, cost/ILP-optimum, the fixed PDNAR primal-dual cost/ILP-optimum,
and episode length for sizes 16, 32, and 64.




