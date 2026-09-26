"""An octree over observed anchors for scene/region pruning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class _Node:
    lower: np.ndarray
    upper: np.ndarray
    indices: np.ndarray
    children: list[_Node] = field(default_factory=list)


class SpatialOctree:
    def __init__(self, anchors, *, bucket_size=16, max_depth=8):
        self.anchors = list(anchors)
        self.positions = np.asarray([a.position_world for a in self.anchors], dtype=float).reshape(
            -1, 3
        )
        self.root = None
        if len(self.positions):
            lower, upper = self.positions.min(axis=0), self.positions.max(axis=0)
            upper = np.maximum(upper, lower + 1e-6)
            self.root = self._build(
                lower, upper, np.arange(len(self.positions)), bucket_size, max_depth
            )

    def _build(self, lower, upper, indices, bucket_size, depth):
        node = _Node(lower, upper, indices)
        if len(indices) <= bucket_size or depth == 0:
            return node
        center = (lower + upper) / 2
        codes = ((self.positions[indices] >= center) * [1, 2, 4]).sum(axis=1)
        for code in range(8):
            subset = indices[codes == code]
            if not len(subset):
                continue
            high = np.array([bool(code & (1 << axis)) for axis in range(3)])
            child_lower = np.where(high, center, lower)
            child_upper = np.where(high, upper, center)
            node.children.append(
                self._build(child_lower, child_upper, subset, bucket_size, depth - 1)
            )
        return node

    def query_box(self, lower, upper):
        lower, upper = np.asarray(lower, dtype=float), np.asarray(upper, dtype=float)
        if (
            lower.shape != (3,)
            or upper.shape != (3,)
            or not np.isfinite([lower, upper]).all()
            or np.any(lower > upper)
        ):
            raise ValueError("region must be finite lower/upper 3D bounds")
        found = []
        pending = [self.root] if self.root is not None else []
        while pending:
            node = pending.pop()
            if np.any(node.upper < lower) or np.any(node.lower > upper):
                continue
            if node.children:
                pending.extend(node.children)
            else:
                inside = np.all(
                    (self.positions[node.indices] >= lower)
                    & (self.positions[node.indices] <= upper),
                    axis=1,
                )
                found.extend(node.indices[inside].tolist())
        return [self.anchors[i] for i in sorted(found)]
