from __future__ import annotations

import numpy as np

from .base import BasePolicy, Decision


class EpsilonGreedyPolicy(BasePolicy):
    """P(best) = 1 - ε + ε/|E|, P(other in E) = ε/|E|,
    где E — exploration-подмножество (по умолчанию = все кандидаты).

    Параметры:
      epsilon           — доля exploration;
      whitelist         — int-индексы, которые разрешено показывать (None = все);
      blacklist         — int-индексы, которые запрещено показывать;
      exploration_subset — int-индексы, среди которых размазывается ε
                           (None = все разрешённые кандидаты).
    """

    name = "epsilon_greedy"

    def __init__(
        self,
        n_actions: int,
        epsilon: float = 0.1,
        seed: int | None = None,
        whitelist: np.ndarray | None = None,
        blacklist: np.ndarray | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, seed=seed)
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError(f"epsilon must be in [0, 1], got {epsilon}")
        self.epsilon = float(epsilon)
        self.whitelist = whitelist
        self.blacklist = blacklist

    def _allowed_mask(self, candidate_items: np.ndarray | None) -> np.ndarray:
        """(n_actions,) boolean mask."""
        if candidate_items is None:
            mask = np.ones(self.n_actions, dtype=bool)
        else:
            mask = np.zeros(self.n_actions, dtype=bool)
            mask[candidate_items] = True
        if self.whitelist is not None:
            wl = np.zeros(self.n_actions, dtype=bool)
            wl[self.whitelist] = True
            mask &= wl
        if self.blacklist is not None:
            bl = np.zeros(self.n_actions, dtype=bool)
            bl[self.blacklist] = True
            mask &= ~bl
        return mask

    def action_distribution(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        # Приводим к 2D
        single_round = base_scores.ndim == 1
        scores = base_scores[None, :] if single_round else base_scores  # (n, A)
        n, n_actions = scores.shape

        allowed = self._allowed_mask(candidate_items)          # (A,)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")

        # argmax только среди allowed
        masked_scores = np.where(allowed[None, :], scores, -np.inf)
        best = np.argmax(masked_scores, axis=1)                # (n,)

        dist = np.zeros((n, n_actions), dtype=float)
        # exploration: равномерно по allowed
        n_allowed = allowed.sum()
        dist[:, allowed] = self.epsilon / n_allowed
        # exploitation: +1-ε на best
        dist[np.arange(n), best] += 1.0 - self.epsilon

        return dist[0] if single_round else dist

    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        dist = self.action_distribution(context, base_scores, candidate_items)
        chosen = int(self.rng.choice(self.n_actions, p=dist))

        allowed = self._allowed_mask(candidate_items)
        greedy = int(np.argmax(np.where(allowed, base_scores, -np.inf)))

        return Decision(
            chosen_item=chosen,
            propensity=float(dist[chosen]),
            probabilities=dist,
            is_exploration=(chosen != greedy),
            policy_name=self.name,
        )

    def update(
        self,
        decision: Decision,
        reward: float,
        context: np.ndarray | None = None,
    ) -> None:
        return None