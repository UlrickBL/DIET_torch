from typing import List, Optional

import torch
import torch.nn as nn


class CRF(nn.Module):

    def __init__(self, num_tags: int):
        super().__init__()
        self.num_tags = num_tags

        # Transition scores: T[i, j] = score of going FROM tag-i TO tag-j
        self.transitions = nn.Parameter(torch.empty(num_tags, num_tags))
        # Score of starting / ending with each tag
        self.start_transitions = nn.Parameter(torch.empty(num_tags))
        self.end_transitions = nn.Parameter(torch.empty(num_tags))

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.uniform_(self.transitions, -0.1, 0.1)
        nn.init.uniform_(self.start_transitions, -0.1, 0.1)
        nn.init.uniform_(self.end_transitions, -0.1, 0.1)

    def forward(
        self,
        emissions: torch.Tensor,               # (B, L, num_tags)
        tags: torch.Tensor,                    # (B, L)  –1 / IGNORE_INDEX OK
        mask: Optional[torch.Tensor] = None,   # (B, L)  bool
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)

        # Replace IGNORE_INDEX with 0 so tensor indexing doesn't blow up;
        # those positions are masked out from the score anyway.
        safe_tags = tags.clone()
        safe_tags[safe_tags == -100] = 0

        ll = self._log_likelihood(emissions, safe_tags, mask)
        return -ll.mean()

    def decode(
        self,
        emissions: torch.Tensor,               # (B, L, num_tags)
        mask: Optional[torch.Tensor] = None,   # (B, L)  bool
    ) -> List[List[int]]:
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        return self._viterbi(emissions, mask)

    def _log_likelihood(
        self,
        emissions: torch.Tensor,   # (B, L, C)
        tags: torch.Tensor,        # (B, L)
        mask: torch.Tensor,        # (B, L)
    ) -> torch.Tensor:             # (B,)
        score = self._sequence_score(emissions, tags, mask)
        partition = self._partition(emissions, mask)
        return score - partition

    def _sequence_score(self, emissions, tags, mask):
        B, L = tags.shape
        # t=0
        score = self.start_transitions[tags[:, 0]]
        score += emissions[:, 0].gather(1, tags[:, 0:1]).squeeze(1)

        for t in range(1, L):
            active = mask[:, t].float()                                  # (B,)
            trans = self.transitions[tags[:, t - 1], tags[:, t]]         # (B,)
            emit = emissions[:, t].gather(1, tags[:, t:t+1]).squeeze(1)  # (B,)
            score += (trans + emit) * active

        # End transitions from the last *real* token
        last_idx = mask.long().sum(1) - 1                                # (B,)
        last_tags = tags.gather(1, last_idx.unsqueeze(1)).squeeze(1)     # (B,)
        score += self.end_transitions[last_tags]
        return score                                                      # (B,)

    def _partition(self, emissions, mask):
        B, L, C = emissions.shape
        # t=0
        score = self.start_transitions.unsqueeze(0) + emissions[:, 0]   # (B, C)

        for t in range(1, L):
            # (B, C, 1) + (1, C, C) → (B, C, C), then logsumexp over prev tag
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)
            next_score = torch.logsumexp(next_score, dim=1)              # (B, C)
            next_score += emissions[:, t]

            # Keep old score for padding positions
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score += self.end_transitions.unsqueeze(0)
        return torch.logsumexp(score, dim=1)                             # (B,)

    def _viterbi(self, emissions, mask):
        B, L, C = emissions.shape
        score = self.start_transitions + emissions[:, 0]                 # (B, C)
        backpointers: List[torch.Tensor] = []

        for t in range(1, L):
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, C, C)
            best_scores, best_prev = next_score.max(dim=1)               # (B, C)
            backpointers.append(best_prev)
            next_score = best_scores + emissions[:, t]
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)

        score += self.end_transitions
        _, best_last = score.max(dim=1)                                  # (B,)

        # Backtrack
        results: List[List[int]] = []
        for b in range(B):
            seq_len = int(mask[b].long().sum().item())
            path = [best_last[b].item()]
            for bp in reversed(backpointers[: seq_len - 1]):
                path.append(bp[b][path[-1]].item())
            path.reverse()
            results.append(path)

        return results
