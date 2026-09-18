from __future__ import annotations

import numpy as np

from .base import BasePolicy, Decision


class LinUCBPolicy(BasePolicy):
    """Linear UCB (Li et al., WWW 2010) со softmax-обёрткой для OPE.

    Состояние для каждого действия a:
        A[a] : (d, d) матрица, инициализируется reg * I
        b[a] : (d,) вектор, инициализируется нулями
        theta[a] = A[a]^{-1} b[a]

    Скор:
        score[a] = theta[a]^T x + alpha * sqrt(x^T A[a]^{-1} x)

    Для OPE используется softmax(score / temperature): это даёт строго
    положительные вероятности для всех действий (common support) и
    корректный propensity выбранного действия.
    """

    name = "linucb"

    def __init__(
        self,
        n_actions: int,
        context_dim: int,
        alpha: float = 1.0,
        temperature: float = 1.0,
        reg: float = 1.0,
        seed: int | None = None,
        whitelist: np.ndarray | None = None,
        blacklist: np.ndarray | None = None,
    ) -> None:
        super().__init__(n_actions=n_actions, seed=seed)
        if context_dim <= 0:
            raise ValueError("context_dim must be positive")
        if alpha < 0:
            raise ValueError("alpha must be >= 0")
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if reg <= 0:
            raise ValueError("reg must be > 0")

        self.context_dim = int(context_dim)
        self.alpha = float(alpha)
        self.temperature = float(temperature)
        self.reg = float(reg)
        self.whitelist = whitelist
        self.blacklist = blacklist

        # Состояние: A[a] = reg * I, b[a] = 0
        self.A = np.stack([reg * np.eye(self.context_dim) for _ in range(self.n_actions)])
        self.b = np.zeros((self.n_actions, self.context_dim))

        # Последний контекст (для update без явной передачи context)
        self._last_context: np.ndarray | None = None

    # ---------------- helpers ----------------

    def _allowed_mask(self, candidate_items: np.ndarray | None) -> np.ndarray:
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

    def _ucb_scores(self, context_batch: np.ndarray) -> np.ndarray:
        """context_batch: (n, d) -> (n, n_actions) UCB-скоров.

        Численно устойчиво: используем np.linalg.solve вместо явного inv,
        где это возможно. Для бонуса нужна квадратичная форма x^T A^{-1} x,
        её считаем через solve(A, x) = A^{-1} x.
        """
        n, d = context_batch.shape
        if d != self.context_dim:
            raise ValueError(
                f"context has dim {d}, policy expects {self.context_dim}"
            )

        scores = np.zeros((n, self.n_actions), dtype=float)
        for a in range(self.n_actions):
            A_a = self.A[a]                                # (d, d)
            b_a = self.b[a]                                # (d,)

            # theta_a = A_a^{-1} b_a
            theta_a = np.linalg.solve(A_a, b_a)            # (d,)

            # mean = X @ theta_a  -> (n,)
            mean = context_batch @ theta_a

            # A_inv_x = A_a^{-1} x для каждого x в батче -> (n, d)
            A_inv_x = np.linalg.solve(A_a, context_batch.T).T

            # bonus = alpha * sqrt(sum(x * A_inv_x, axis=1))
            quad = np.einsum("nd,nd->n", context_batch, A_inv_x)
            bonus = self.alpha * np.sqrt(np.maximum(quad, 0.0))  # защита от -eps

            scores[:, a] = mean + bonus
        return scores

    def _softmax(self, scores: np.ndarray) -> np.ndarray:
        z = scores / self.temperature
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    # ---------------- BasePolicy API ----------------

    def action_distribution(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> np.ndarray:
        single_round = context.ndim == 1
        x = context[None, :] if single_round else context       # (n, d)

        ucb = self._ucb_scores(x)                                # (n, A)

        # маскирование кандидатов: -inf для не-кандидатов
        allowed = self._allowed_mask(candidate_items)            # (A,)
        if not allowed.any():
            raise ValueError("No allowed candidates for this policy")
        ucb = np.where(allowed[None, :], ucb, -np.inf)

        dist = self._softmax(ucb)                                # (n, A)
        return dist[0] if single_round else dist

    def select_action(
        self,
        context: np.ndarray,
        base_scores: np.ndarray,
        candidate_items: np.ndarray | None = None,
    ) -> Decision:
        dist = self.action_distribution(context, base_scores, candidate_items)

        # запоминаем контекст для update без явного context
        self._last_context = np.array(context, copy=True)

        chosen = int(self.rng.choice(self.n_actions, p=dist))

        # определение exploration: выбранное действие != argmax UCB среди кандидатов
        allowed = self._allowed_mask(candidate_items)
        ucb = self._ucb_scores(context[None, :] if context.ndim == 1 else context)
        ucb_masked = np.where(allowed[None, :], ucb, -np.inf)
        greedy = int(np.argmax(ucb_masked, axis=1)[0])

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
        """Обновление состояния для выбранного действия.

        Если context не передан, используется последний контекст из
        select_action (работает для пошагового режима).
        """
        if context is None:
            if self._last_context is None:
                raise ValueError(
                    "context must be provided to update() if select_action "
                    "was not called on this policy"
                )
            context = self._last_context

        x = np.asarray(context, dtype=float).reshape(-1)
        if x.shape[0] != self.context_dim:
            raise ValueError(
                f"context has dim {x.shape[0]}, policy expects {self.context_dim}"
            )

        a = decision.chosen_item
        self.A[a] += np.outer(x, x)
        self.b[a] += float(reward) * x