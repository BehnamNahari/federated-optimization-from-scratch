import copy
import random

import torch


class Server:

    def __init__(
        self,
        model,
        clients,
        client_fraction=1.0,
        seed=42
    ):
        self.global_model = model
        self.clients = clients
        self.client_fraction = client_fraction

        self.rng = random.Random(seed)

    def select_clients(self):

        num_clients = len(self.clients)

        num_selected = max(
            1,
            int(self.client_fraction * num_clients)
        )

        selected_ids = self.rng.sample(
            list(self.clients.keys()),
            num_selected
        )

        return selected_ids

    def aggregate(self, local_states, client_sizes):

        total_samples = sum(client_sizes)

        global_state = copy.deepcopy(
            local_states[0]
        )

        for key in global_state:

            global_state[key] = torch.zeros_like(
                global_state[key]
            )

        for local_state, client_size in zip(
            local_states,
            client_sizes
        ):

            weight = client_size / total_samples

            for key in global_state:

                global_state[key] += (
                    weight * local_state[key]
                )

        self.global_model.load_state_dict(
            global_state
        )