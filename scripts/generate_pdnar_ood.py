#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
import random
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from scipy.optimize import Bounds, LinearConstraint, milp
from torch_geometric.data import Data


# ---------------------------------------------------------------------------
# PDNAR-compatible bipartite BA generator
# ---------------------------------------------------------------------------

def generate_pdnar_bipartite_graph(
    num_nodes: int,
    b: int,
    rng: random.Random,
) -> tuple[nx.Graph, int]:
    """
    Reproduce the distribution used by PDNAR's released implementation.

    num_nodes = total number of bipartite nodes.

    The number of left-side (element) nodes is sampled uniformly from
    [num_nodes//4, 3*num_nodes//4].

    Each right-side (set) node attaches to b left-side nodes using
    preferential attachment.

    IMPORTANT:
    PDNAR's released implementation uses random.choices(), i.e. sampling
    WITH replacement. We reproduce that behavior here for comparability
    with the released implementation rather than silently correcting it.
    """

    n_left = max(
        3,
        rng.randint(
            num_nodes // 4,
            (num_nodes // 4) * 3,
        ),
    )

    n_right = num_nodes - n_left

    if n_right <= 0:
        raise ValueError(
            f"Invalid partition for num_nodes={num_nodes}: "
            f"n_left={n_left}, n_right={n_right}"
        )

    graph = nx.Graph()

    left_nodes = list(range(n_left))
    right_nodes = list(range(n_left, num_nodes))

    graph.add_nodes_from(
        left_nodes,
        bipartite=1,
    )

    for right in right_nodes:
        graph.add_node(
            right,
            bipartite=0,
        )

        probabilities = [
            graph.degree(left) + 1
            for left in left_nodes
        ]

        total = sum(probabilities)

        probabilities = [
            p / total
            for p in probabilities
        ]

        selected = rng.choices(
            left_nodes,
            weights=probabilities,
            k=b,
        )

        graph.add_edges_from(
            (right, left)
            for left in selected
        )

    weights = [
        rng.uniform(1e-10, 1.0)
        for _ in range(n_left)
    ]

    graph.graph["weights"] = weights

    return graph, n_left


# ---------------------------------------------------------------------------
# Exact MSC optimum
# ---------------------------------------------------------------------------

def solve_msc(
    graph: nx.Graph,
    n_left: int,
):
    """
    Solve MSC exactly using the same formulation as PDNAR.

    PDNAR's representation is:

        left nodes  = candidate sets
        right nodes = elements that must be covered
        weights     = costs of left/set nodes

    Therefore:
        - one binary variable per left node
        - one covering constraint per right node

    This intentionally matches:
        dransyhe/pdnar/src/dataset/algorithms/set_cover_solver.py
    """

    num_total = graph.number_of_nodes()
    n_right = num_total - n_left

    left_nodes = list(range(n_left))
    right_nodes = list(range(n_left, num_total))

    weights = np.asarray(
        graph.graph["weights"],
        dtype=np.float64,
    )

    if len(weights) != n_left:
        raise ValueError(
            f"Expected {n_left} set weights, "
            f"got {len(weights)}."
        )

    # ---------------------------------------------------------------
    # A[e, s] = 1 iff set s covers element e.
    #
    # This is exactly the orientation used by PDNAR:
    #
    #   A.shape = (num_right_nodes, num_left_nodes)
    #
    # and the constraints are:
    #
    #   A x >= 1
    # ---------------------------------------------------------------

    A = np.zeros(
        (n_right, n_left),
        dtype=np.float64,
    )

    for e_idx, right in enumerate(right_nodes):

        neighbors = list(
            graph.neighbors(right)
        )

        if not neighbors:
            raise RuntimeError(
                "Generated an uncovered element: "
                f"right node {right} has no neighboring set."
            )

        for left in neighbors:
            A[e_idx, left] = 1.0

    constraints = LinearConstraint(
        A,
        lb=np.ones(n_right),
        ub=np.full(
            n_right,
            np.inf,
        ),
    )

    result = milp(
        c=weights,
        integrality=np.ones(
            n_left,
            dtype=np.int8,
        ),
        bounds=Bounds(
            np.zeros(n_left),
            np.ones(n_left),
        ),
        constraints=constraints,
        options={
            "presolve": True,
        },
    )

    if not result.success:
        raise RuntimeError(
            "HiGHS failed to solve MSC: "
            f"{result.message}"
        )

    solution = np.rint(
        result.x
    ).astype(np.int64)

    objective = float(
        weights @ solution
    )

    return solution, objective, A


# ---------------------------------------------------------------------------
# PDNAR primal-dual trajectory
# ---------------------------------------------------------------------------
def primal_dual_msc(
    graph: nx.Graph,
    n_left: int,
):
    """
    Reproduce PDNAR's set-cover primal-dual trajectory.

    This intentionally follows:
        dransyhe/pdnar/src/dataset/algorithms/set_cover.py

    Left nodes:
        0 .. n_left-1

    Right nodes:
        n_left .. n_left+n_right-1

    PDNAR mutates the working graph. When a primal node is deleted,
    its neighboring right nodes are also removed from the graph.
    We reproduce that behavior exactly.
    """

    B = graph.copy()

    num_bipartite_nodes = B.number_of_nodes()
    num_edges = num_bipartite_nodes - n_left

    weights = np.asarray(
        B.graph["weights"],
        dtype=np.float64,
    )

    if len(weights) != n_left:
        raise ValueError(
            f"Expected {n_left} weights, "
            f"got {len(weights)}."
        )

    w_p = weights.copy()

    node_mask = [
        False
        for _ in range(n_left)
    ]

    edge_mask = [
        True
        for _ in range(num_edges)
    ]

    # PDNAR initialization.
    all_delta = [
        [0.0 for _ in range(num_edges)]
    ]

    all_w_p = [
        w_p.copy()
    ]

    all_node_mask = [
        node_mask.copy()
    ]

    all_edge_mask = [
        edge_mask.copy()
    ]

    # ------------------------------------------------------------------
    # Exact PDNAR loop
    # ------------------------------------------------------------------

    while any(edge_mask):

        del_nodes = []

        delta = [
            0.0
            for _ in range(num_edges)
        ]

        # Calculate degree for every currently active set node.
        d_p = [
            0.0
            for _ in range(n_left)
        ]

        for n in range(n_left):

            if node_mask[n]:
                continue

            d_p[n] = B.degree(n)

        # Calculate dual increment for every remaining element.
        for e in range(num_edges):

            if not edge_mask[e]:
                continue

            min_delta = float("inf")

            right_node = n_left + e

            # This is exactly the PDNAR neighbor traversal.
            if right_node in B:
                for v in B.neighbors(
                    right_node
                ):

                    if d_p[v] <= 0:
                        continue

                    min_delta = min(
                        min_delta,
                        w_p[v] / d_p[v],
                    )

            if min_delta == float("inf"):
                min_delta = 0.0

            delta[e] = min_delta

        # Update residual weights.
        for n in range(n_left):

            if node_mask[n]:
                continue

            reduction = 0.0

            if n in B:
                for right in B.neighbors(n):

                    e = right - n_left

                    if (
                        0 <= e < num_edges
                        and edge_mask[e]
                    ):
                        reduction += delta[e]

            w_p[n] -= reduction

            if (
                w_p[n]
                <= weights[n] * 0.1
            ):
                del_nodes.append(n)

        # ------------------------------------------------------------------
        # CRITICAL:
        #
        # PDNAR removes BOTH:
        #   1. the right/element nodes adjacent to the selected set
        #   2. the selected set node itself
        #
        # We must reproduce this graph mutation.
        # ------------------------------------------------------------------

        for n in del_nodes:

            if n not in B:
                continue

            neighbors = list(
                B.neighbors(n)
            )

            for right in neighbors:

                e = right - n_left

                if (
                    0 <= e < num_edges
                ):
                    edge_mask[e] = False

                # Exact PDNAR behavior:
                # remove the entire right/element node.
                if right in B:
                    B.remove_node(right)

            # Exact PDNAR behavior:
            # remove the selected left/set node.
            B.remove_node(n)

            node_mask[n] = True

        # Record trajectory.
        all_delta.append(
            delta.copy()
        )

        all_w_p.append(
            w_p.copy()
        )

        all_node_mask.append(
            node_mask.copy()
        )

        all_edge_mask.append(
            edge_mask.copy()
        )

        # A valid PDNAR iteration must delete at least one
        # primal node whenever active elements remain.
        if not del_nodes:
            active_elements = [
                e
                for e in range(num_edges)
                if edge_mask[e]
            ]

            raise RuntimeError(
                "PDNAR primal-dual algorithm made no progress. "
                f"Active elements: {active_elements}; "
                f"residual weights: {w_p.tolist()}"
            )

    # ------------------------------------------------------------------
    # PDNAR pads trajectories to the number of bipartite nodes.
    # ------------------------------------------------------------------

    timesteps = len(all_delta)

    if timesteps > num_bipartite_nodes:
        raise RuntimeError(
            "PDNAR trajectory has more timesteps than "
            "the number of bipartite nodes: "
            f"{timesteps} > {num_bipartite_nodes}"
        )

    to_add = (
        num_bipartite_nodes
        - timesteps
    )

    for _ in range(to_add):

        all_delta.append(
            all_delta[-1].copy()
        )

        all_w_p.append(
            all_w_p[-1].copy()
        )

        all_node_mask.append(
            all_node_mask[-1].copy()
        )

        all_edge_mask.append(
            all_edge_mask[-1].copy()
        )

    return (
        all_delta,
        all_w_p,
        all_node_mask,
        all_edge_mask,
    )

# ---------------------------------------------------------------------------
# Convert one graph to the serialized PDNAR record format
# ---------------------------------------------------------------------------

def make_record(
    graph: nx.Graph,
    n_left: int,
):
    original = graph.copy()

    optimal_solution, optimal_weight, A = solve_msc(
        original,
        n_left,
    )

    (
        all_delta,
        all_w_p,
        all_node_mask,
        all_edge_mask,
    ) = primal_dual_msc(
        graph,
        n_left,
    )

    num_total = original.number_of_nodes()
    num_edges = num_total - n_left

    # Construct edge_index in exactly the orientation used by PDNAR.
    edge_pairs = []

    for e in range(num_edges):

        right = n_left + e

        for left in original.neighbors(right):
            edge_pairs.append(
                (e, left)
            )

    if edge_pairs:
        edge_index = torch.tensor(
            edge_pairs,
            dtype=torch.long,
        ).t().contiguous()
    else:
        edge_index = torch.empty(
            (2, 0),
            dtype=torch.long,
        )

    delta = torch.tensor(
        np.asarray(all_delta),
        dtype=torch.float32,
    ).transpose(
        0,
        1,
    ).unsqueeze(-1)

    w_p = torch.tensor(
        np.asarray(all_w_p),
        dtype=torch.float32,
    ).transpose(
        0,
        1,
    ).unsqueeze(-1)

    node_mask = torch.tensor(
        np.asarray(all_node_mask),
        dtype=torch.float32,
    ).transpose(
        0,
        1,
    ).unsqueeze(-1)

    edge_mask = torch.tensor(
        np.asarray(all_edge_mask),
        dtype=torch.float32,
    ).transpose(
        0,
        1,
    ).unsqueeze(-1)

    record = Data(
        x=w_p,
        y=delta,
        x_mask=node_mask,
        y_mask=edge_mask,
        edge_index=edge_index,
        primal_optimal_solution=torch.tensor(
            optimal_solution,
            dtype=torch.long,
        ),
        primal_optimal_weight=torch.tensor(
            [optimal_weight],
            dtype=torch.float32,
        ),
    )

    return record


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_dataset(
    output_root: Path,
    sizes: list[int],
    b_values: list[int],
    samples: int,
    seeds: int,
    base_seed: int,
):
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    for b in b_values:

        for size in sizes:

            for seed_idx in range(seeds):

                seed = (
                    base_seed
                    + 100000 * b
                    + 1000 * size
                    + seed_idx
                )

                rng = random.Random(seed)

                records = []

                print(
                    f"\n"
                    f"Generating "
                    f"b={b}, "
                    f"n={size}, "
                    f"seed={seed_idx + 1}/{seeds}"
                )

                for i in range(samples):

                    graph, n_left = (
                        generate_pdnar_bipartite_graph(
                            size,
                            b,
                            rng,
                        )
                    )

                    record = make_record(
                        graph,
                        n_left,
                    )

                    records.append(
                        record
                    )

                    if (
                        (i + 1) % max(
                            1,
                            samples // 10,
                        ) == 0
                    ):
                        print(
                            f"\r"
                            f"  "
                            f"{100.0 * (i + 1) / samples:6.2f}%",
                            end="",
                            flush=True,
                        )

                print()

                output_dir = (
                    output_root
                    / f"b_{b}"
                    / f"n_{size}"
                )

                output_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                path = (
                    output_dir
                    / f"test_{samples}_seed_{seed_idx}.pkl"
                )

                with path.open("wb") as f:
                    pickle.dump(
                        records,
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )

                print(
                    f"  saved: {path}"
                )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate PDNAR-compatible MSC OOD "
            "test datasets."
        )
    )

    parser.add_argument(
        "--output-root",
        default="dataset/pdnar_ood",
    )

    parser.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[
            16,
            32,
            64,
            128,
            256,
            512,
            1024,
        ],
    )

    parser.add_argument(
        "--b",
        nargs="+",
        type=int,
        default=[
            3,
            8,
        ],
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    generate_dataset(
        output_root=Path(
            args.output_root
        ),
        sizes=args.sizes,
        b_values=args.b,
        samples=args.samples,
        seeds=args.seeds,
        base_seed=args.seed,
    )


if __name__ == "__main__":
    main()