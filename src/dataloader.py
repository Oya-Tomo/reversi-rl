import torch
from torch.utils.data import Dataset

from agent import ExperienceBuffer


def exps_to_data(
    exps: list[ExperienceBuffer],
    tail: int,
) -> list[
    tuple[
        torch.Tensor | None,
        float,
        torch.Tensor | None,
    ]
]:
    data = []

    for exp in exps:
        exp_buffer = exp.buffer[max(len(exp.buffer) - tail, 0) :]
        for i in range(len(exp_buffer)):
            if i == len(exp_buffer) - 1:
                data.append(
                    (
                        torch.tensor(exp_buffer[i].inputs, dtype=torch.float),
                        exp_buffer[i].reward,
                        None,
                    )
                )
            else:
                data.append(
                    (
                        torch.tensor(exp_buffer[i].inputs, dtype=torch.float),
                        exp_buffer[i].reward,
                        torch.tensor(exp_buffer[i + 1].inputs, dtype=torch.float),
                    )
                )
    return data


class QvalueDataset(Dataset):
    def __init__(
        self,
        exps: list[ExperienceBuffer],
        tail: int,
        model: torch.nn.Module,
    ) -> None:
        self.data = exps_to_data(exps, tail)
        self.model = model
        self.device = next(model.parameters()).device

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        self.model.eval()
        item = self.data[idx]

        target_inputs = item[2]
        if target_inputs != None:
            target_outputs: torch.Tensor = self.model(
                target_inputs.to(self.device)
            ).cpu()
            return item[0], item[1] + target_outputs
        else:
            return item[0], torch.tensor([item[1]])
