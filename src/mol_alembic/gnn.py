"""Pure-NumPy Graph Attention and Message Passing Neural Network with analytical backpropagation."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from numpy.typing import NDArray

from .featurizer import ATOM_FEATURE_DIM, EDGE_FEATURE_DIM, featurize_molecule
from .smiles import parse_smiles
from .types import GraphShapeError, GraphTensors


def _silu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Smooth SiLU (Swish) activation: x * sigmoid(x)."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    return x * sig


def _silu_grad(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Exact derivative of SiLU(x)."""
    sig = 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))
    return sig * (1.0 + x * (1.0 - sig))


def _leaky_relu(x: NDArray[np.float64], alpha: float = 0.2) -> NDArray[np.float64]:
    return np.where(x >= 0.0, x, alpha * x)


def _leaky_relu_grad(x: NDArray[np.float64], alpha: float = 0.2) -> NDArray[np.float64]:
    return np.where(x >= 0.0, 1.0, alpha)


@dataclass
class GNNConfig:
    """Hyperparameters for MolecularGraphNet."""

    node_dim: int = ATOM_FEATURE_DIM
    edge_dim: int = EDGE_FEATURE_DIM
    hidden_dim: int = 32
    num_layers: int = 3
    learning_rate: float = 0.008
    weight_decay: float = 1e-4
    dropout_rate: float = 0.08
    seed: int = 42


@dataclass
class _ForwardCache:
    """Cached intermediates for exact analytical backpropagation on a single molecule graph."""

    graph: GraphTensors
    h_in: NDArray[np.float64]
    z_in: NDArray[np.float64]
    layer_h: List[NDArray[np.float64]]
    layer_msg_raw: List[NDArray[np.float64]]
    layer_agg: List[NDArray[np.float64]]
    layer_pre_act: List[NDArray[np.float64]]
    layer_attn_scores: List[NDArray[np.float64]]
    layer_attn_weights: List[NDArray[np.float64]]
    dropout_masks: List[NDArray[np.float64]]
    gate_logits: NDArray[np.float64]
    gate_weights: NDArray[np.float64]
    pooled: NDArray[np.float64]
    head_pre: NDArray[np.float64]
    head_act: NDArray[np.float64]
    pred: float


@dataclass
class MolecularGraphNet:
    """Edge-conditioned Message Passing + Graph Attention Network with gated global pooling."""

    config: GNNConfig = field(default_factory=GNNConfig)
    params: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    adam_m: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    adam_v: Dict[str, NDArray[np.float64]] = field(default_factory=dict)
    adam_step: int = 0
    y_mean: float = 0.0
    y_std: float = 1.0
    training_history: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.params:
            self._init_weights()

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.config.seed)
        d_in = self.config.node_dim
        d_e = self.config.edge_dim
        h = self.config.hidden_dim

        def glorot(r: int, c: int) -> NDArray[np.float64]:
            scale = math.sqrt(2.0 / float(r + c))
            return rng.normal(0.0, scale, size=(r, c))

        self.params = {
            "W_in": glorot(d_in, h),
            "b_in": np.zeros(h, dtype=np.float64),
            "W_gate": glorot(h, 1),
            "b_gate": np.zeros(1, dtype=np.float64),
            "W_head1": glorot(h, h),
            "b_head1": np.zeros(h, dtype=np.float64),
            "W_head2": glorot(h, 1),
            "b_head2": np.zeros(1, dtype=np.float64),
        }
        for l_idx in range(self.config.num_layers):
            self.params[f"W_self_{l_idx}"] = glorot(h, h)
            self.params[f"W_nbr_{l_idx}"] = glorot(h, h)
            self.params[f"W_edge_{l_idx}"] = glorot(d_e, h)
            self.params[f"b_msg_{l_idx}"] = np.zeros(h, dtype=np.float64)
            self.params[f"a_attn_{l_idx}"] = rng.normal(0.0, 0.15, size=(2 * h + d_e, 1))

        self.adam_m = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_v = {k: np.zeros_like(v) for k, v in self.params.items()}
        self.adam_step = 0

    def _forward_single(
        self,
        graph: GraphTensors,
        training: bool = False,
        rng: Optional[np.random.Generator] = None,
    ) -> _ForwardCache:
        n = graph.node_features.shape[0]
        if n == 0:
            raise GraphShapeError("Empty graph cannot be passed through MolecularGraphNet")

        h = self.config.hidden_dim
        z_in = graph.node_features @ self.params["W_in"] + self.params["b_in"]
        h_cur = _silu(z_in)

        layer_h: List[NDArray[np.float64]] = [h_cur]
        layer_msg_raw: List[NDArray[np.float64]] = []
        layer_agg: List[NDArray[np.float64]] = []
        layer_pre_act: List[NDArray[np.float64]] = []
        layer_attn_scores: List[NDArray[np.float64]] = []
        layer_attn_weights: List[NDArray[np.float64]] = []
        dropout_masks: List[NDArray[np.float64]] = []

        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        num_e = graph.edge_index.shape[1]

        for l_idx in range(self.config.num_layers):
            w_self = self.params[f"W_self_{l_idx}"]
            w_nbr = self.params[f"W_nbr_{l_idx}"]
            w_edge = self.params[f"W_edge_{l_idx}"]
            b_msg = self.params[f"b_msg_{l_idx}"]
            a_attn = self.params[f"a_attn_{l_idx}"]

            agg = np.zeros((n, h), dtype=np.float64)
            if num_e > 0:
                h_src = h_cur[src]
                h_dst = h_cur[dst]
                msg_raw = h_src @ w_nbr + graph.edge_features @ w_edge + b_msg
                cat = np.hstack([h_dst, h_src, graph.edge_features])
                attn_pre = cat @ a_attn
                attn_lrelu = _leaky_relu(attn_pre).ravel()

                attn_w = np.zeros(num_e, dtype=np.float64)
                for node_i in range(n):
                    mask_i = dst == node_i
                    if np.any(mask_i):
                        logits_i = attn_lrelu[mask_i]
                        logits_i = logits_i - np.max(logits_i)
                        exp_i = np.exp(logits_i)
                        attn_w[mask_i] = exp_i / np.maximum(np.sum(exp_i), 1e-12)

                weighted_msg = attn_w[:, None] * msg_raw
                np.add.at(agg, dst, weighted_msg)
            else:
                msg_raw = np.zeros((0, h), dtype=np.float64)
                attn_pre = np.zeros((0, 1), dtype=np.float64)
                attn_w = np.zeros(0, dtype=np.float64)

            spec_agg = graph.norm_adjacency @ h_cur
            pre_act = h_cur @ w_self + 0.65 * agg + 0.35 * (spec_agg @ w_nbr)
            act = _silu(pre_act)

            if training and self.config.dropout_rate > 0.0 and rng is not None:
                keep_prob = 1.0 - self.config.dropout_rate
                d_mask = (rng.random(act.shape) < keep_prob).astype(np.float64) / keep_prob
            else:
                d_mask = np.ones_like(act)

            h_next = h_cur + act * d_mask
            layer_msg_raw.append(msg_raw)
            layer_agg.append(agg)
            layer_pre_act.append(pre_act)
            layer_attn_scores.append(attn_pre)
            layer_attn_weights.append(attn_w)
            dropout_masks.append(d_mask)
            h_cur = h_next
            layer_h.append(h_cur)

        gate_logits = (h_cur @ self.params["W_gate"] + self.params["b_gate"]).ravel()
        g_shift = gate_logits - np.max(gate_logits)
        g_exp = np.exp(g_shift)
        gate_w = g_exp / np.maximum(np.sum(g_exp), 1e-12)
        pooled = np.sum(gate_w[:, None] * h_cur, axis=0)

        head_pre = pooled @ self.params["W_head1"] + self.params["b_head1"]
        head_act = _silu(head_pre)
        pred_norm = float((head_act @ self.params["W_head2"] + self.params["b_head2"])[0])

        return _ForwardCache(
            graph=graph,
            h_in=h_cur,
            z_in=z_in,
            layer_h=layer_h,
            layer_msg_raw=layer_msg_raw,
            layer_agg=layer_agg,
            layer_pre_act=layer_pre_act,
            layer_attn_scores=layer_attn_scores,
            layer_attn_weights=layer_attn_weights,
            dropout_masks=dropout_masks,
            gate_logits=gate_logits,
            gate_weights=gate_w,
            pooled=pooled,
            head_pre=head_pre,
            head_act=head_act,
            pred=pred_norm,
        )

    def _backward_single(
        self,
        cache: _ForwardCache,
        target_norm: float,
    ) -> Tuple[float, Dict[str, NDArray[np.float64]]]:
        """Compute MSE loss and analytical parameter gradients for one molecule graph."""
        grads = {k: np.zeros_like(v) for k, v in self.params.items()}
        err = cache.pred - target_norm
        loss = 0.5 * err * err
        d_pred = err

        grads["W_head2"] += np.outer(cache.head_act, np.array([d_pred]))
        grads["b_head2"] += np.array([d_pred])
        d_head_act = d_pred * self.params["W_head2"].ravel()
        d_head_pre = d_head_act * _silu_grad(cache.head_pre)

        grads["W_head1"] += np.outer(cache.pooled, d_head_pre)
        grads["b_head1"] += d_head_pre
        d_pooled = self.params["W_head1"] @ d_head_pre

        h_final = cache.layer_h[-1]
        n = h_final.shape[0]
        gw = cache.gate_weights
        d_h = gw[:, None] * d_pooled[None, :]
        d_gw = np.sum(h_final * d_pooled[None, :], axis=1)
        dot_gw = float(np.sum(gw * d_gw))
        d_glogits = gw * (d_gw - dot_gw)

        grads["W_gate"] += h_final.T @ d_glogits[:, None]
        grads["b_gate"] += np.array([float(np.sum(d_glogits))])
        d_h += np.outer(d_glogits, self.params["W_gate"].ravel())

        graph = cache.graph
        src = graph.edge_index[0]
        dst = graph.edge_index[1]
        num_e = graph.edge_index.shape[1]

        for l_idx in reversed(range(self.config.num_layers)):
            h_prev = cache.layer_h[l_idx]
            pre_act = cache.layer_pre_act[l_idx]
            d_mask = cache.dropout_masks[l_idx]
            attn_w = cache.layer_attn_weights[l_idx]
            attn_pre = cache.layer_attn_scores[l_idx]
            msg_raw = cache.layer_msg_raw[l_idx]

            d_act = d_h * d_mask
            d_pre = d_act * _silu_grad(pre_act)
            d_h_prev = d_h.copy()

            w_self = self.params[f"W_self_{l_idx}"]
            w_nbr = self.params[f"W_nbr_{l_idx}"]
            a_attn = self.params[f"a_attn_{l_idx}"]
            h_dim = self.config.hidden_dim

            grads[f"W_self_{l_idx}"] += h_prev.T @ d_pre
            d_h_prev += d_pre @ w_self.T

            d_spec = 0.35 * d_pre
            spec_h = graph.norm_adjacency @ h_prev
            grads[f"W_nbr_{l_idx}"] += spec_h.T @ d_spec
            d_h_prev += graph.norm_adjacency.T @ (d_spec @ w_nbr.T)

            if num_e > 0:
                d_agg = 0.65 * d_pre
                d_weighted_msg = d_agg[dst]
                d_msg_raw = attn_w[:, None] * d_weighted_msg
                d_attn_w = np.sum(d_weighted_msg * msg_raw, axis=1)

                d_attn_lrelu = np.zeros(num_e, dtype=np.float64)
                for node_i in range(n):
                    mask_i = dst == node_i
                    if np.any(mask_i):
                        w_i = attn_w[mask_i]
                        dw_i = d_attn_w[mask_i]
                        dot_i = float(np.sum(w_i * dw_i))
                        d_attn_lrelu[mask_i] = w_i * (dw_i - dot_i)

                d_attn_pre = d_attn_lrelu[:, None] * _leaky_relu_grad(attn_pre)
                h_src = h_prev[src]
                h_dst = h_prev[dst]
                cat = np.hstack([h_dst, h_src, graph.edge_features])
                grads[f"a_attn_{l_idx}"] += cat.T @ d_attn_pre

                d_cat = d_attn_pre @ a_attn.T
                d_h_dst_attn = d_cat[:, :h_dim]
                d_h_src_attn = d_cat[:, h_dim : 2 * h_dim]
                np.add.at(d_h_prev, dst, d_h_dst_attn)
                np.add.at(d_h_prev, src, d_h_src_attn)

                grads[f"W_nbr_{l_idx}"] += h_src.T @ d_msg_raw
                grads[f"W_edge_{l_idx}"] += graph.edge_features.T @ d_msg_raw
                grads[f"b_msg_{l_idx}"] += np.sum(d_msg_raw, axis=0)
                np.add.at(d_h_prev, src, d_msg_raw @ w_nbr.T)

            d_h = d_h_prev

        d_zin = d_h * _silu_grad(cache.z_in)
        grads["W_in"] += graph.node_features.T @ d_zin
        grads["b_in"] += np.sum(d_zin, axis=0)
        return loss, grads

    def _adamw_step(self, grads: Dict[str, NDArray[np.float64]], lr: float) -> None:
        self.adam_step += 1
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        wd = self.config.weight_decay

        total_sq = sum(float(np.sum(g * g)) for g in grads.values())
        grad_norm = math.sqrt(total_sq)
        clip_coef = min(1.0, 5.0 / max(1e-12, grad_norm))

        for name, param in self.params.items():
            g = grads[name] * clip_coef
            m = self.adam_m[name]
            v = self.adam_v[name]
            m[:] = beta1 * m + (1.0 - beta1) * g
            v[:] = beta2 * v + (1.0 - beta2) * (g * g)
            m_hat = m / (1.0 - beta1**self.adam_step)
            v_hat = v / (1.0 - beta2**self.adam_step)
            if param.ndim == 2:
                param *= 1.0 - lr * wd
            param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    def fit(
        self,
        smiles_list: Sequence[str],
        targets: Sequence[float],
        epochs: int = 35,
        batch_size: int = 8,
    ) -> List[float]:
        """Train the GAT-MPNN on SMILES strings and scalar property targets."""
        if len(smiles_list) != len(targets) or not smiles_list:
            raise GraphShapeError("Training SMILES and targets must be non-empty and equal length")

        graphs = [featurize_molecule(parse_smiles(s, name=f"mol_{i}")) for i, s in enumerate(smiles_list)]
        return self.fit_graphs(graphs, targets, epochs=epochs, batch_size=batch_size)

    def fit_graphs(
        self,
        graphs: Sequence[GraphTensors],
        targets: Sequence[float],
        epochs: int = 35,
        batch_size: int = 8,
    ) -> List[float]:
        """Train directly on pre-featurized GraphTensors."""
        y_arr = np.asarray(targets, dtype=np.float64)
        self.y_mean = float(np.mean(y_arr))
        std_raw = float(np.std(y_arr))
        self.y_std = std_raw if std_raw > 1e-6 else 1.0
        y_norm = (y_arr - self.y_mean) / self.y_std

        rng = np.random.default_rng(self.config.seed)
        n = len(graphs)
        history: List[float] = []

        for epoch in range(epochs):
            cos_factor = 0.5 * (1.0 + math.cos(math.pi * epoch / max(1, epochs)))
            lr = self.config.learning_rate * (0.15 + 0.85 * cos_factor)

            perm = rng.permutation(n)
            epoch_loss = 0.0
            for start in range(0, n, batch_size):
                batch_ids = perm[start : start + batch_size]
                acc_grads = {k: np.zeros_like(v) for k, v in self.params.items()}
                for idx in batch_ids:
                    cache = self._forward_single(graphs[int(idx)], training=True, rng=rng)
                    loss_i, g_i = self._backward_single(cache, float(y_norm[int(idx)]))
                    epoch_loss += loss_i
                    for k in acc_grads:
                        acc_grads[k] += g_i[k]
                inv_b = 1.0 / float(len(batch_ids))
                for k in acc_grads:
                    acc_grads[k] *= inv_b
                self._adamw_step(acc_grads, lr=lr)

            history.append(epoch_loss / float(n))

        self.training_history = history
        return history

    def predict_graph(
        self,
        graph: GraphTensors,
        mc_samples: int = 12,
    ) -> Tuple[float, float, NDArray[np.float64]]:
        """Predict property value, MC-dropout uncertainty, and atom-level attention saliency."""
        det_cache = self._forward_single(graph, training=False)
        pred = det_cache.pred * self.y_std + self.y_mean
        atom_saliency = det_cache.gate_weights.copy()

        if mc_samples <= 1 or self.config.dropout_rate <= 0.0:
            return pred, 0.05 * self.y_std, atom_saliency

        rng = np.random.default_rng(self.config.seed + graph.node_features.shape[0])
        samples: List[float] = []
        for _ in range(mc_samples):
            mc_cache = self._forward_single(graph, training=True, rng=rng)
            samples.append(mc_cache.pred * self.y_std + self.y_mean)
        unc = float(np.std(samples, ddof=1)) if len(samples) > 1 else 0.0
        return pred, max(1e-4, unc), atom_saliency

    def predict_smiles(self, smiles: str, mc_samples: int = 12) -> Tuple[float, float, NDArray[np.float64]]:
        """Featurize SMILES and return (prediction, uncertainty, atom_saliency)."""
        graph = featurize_molecule(parse_smiles(smiles))
        return self.predict_graph(graph, mc_samples=mc_samples)
