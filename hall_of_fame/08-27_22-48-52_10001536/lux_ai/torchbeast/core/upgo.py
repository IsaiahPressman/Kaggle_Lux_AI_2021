import collections
import torch

UPGOReturns = collections.namedtuple("UPGOReturns", "vs advantages")


@torch.no_grad()
def upgo(
        rewards: torch.Tensor,
        values: torch.Tensor,
        bootstrap_value: torch.Tensor,
        discounts: torch.Tensor,
        lmb: float,
) -> UPGOReturns:
    # Append bootstrapped value to get [v1, ..., v_t+1]
    values_t_plus_1 = torch.cat(
        [values[1:], torch.unsqueeze(bootstrap_value, 0)], dim=0
    )
    target_values = [bootstrap_value]
    for t in range(discounts.shape[0] - 1, -1, -1):
        # noinspection PyUnresolvedReferences
        target_values.append(
            rewards[t] + discounts[t] * torch.max(values_t_plus_1[t],
                                                  (1 - lmb) * values_t_plus_1[t] + lmb * target_values[-1])
        )
    target_values.reverse()
    # Remove bootstrap value from end of target_values list
    target_values = torch.stack(target_values[:-1], dim=0)

    return UPGOReturns(
        vs=target_values,
        advantages=target_values - values
    )
