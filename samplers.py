import torch
import random

from rdkit import Chem
from typing import Optional
from itertools import chain
from collections import defaultdict

from torch.utils.data import Sampler, DistributedSampler
from torch_geometric.loader import DataLoader

from .utils import boltzmann_average
from data.drugs import Drugs


class EnsembleSampler:
    def __init__(self, dataset, batch_size, strategy='all', shuffle=True, node_limit=2000, batch_node_limit=5000):
        assert strategy in ['all', 'random', 'first']
        self.molecule_idx = dataset.data.molecule_idx[dataset._indices]
        self.num_nodes = [d.num_nodes for d in dataset]
        self.num_molecules = dataset.num_molecules
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle = shuffle
        self.node_limit = node_limit
        self.batch_node_limit = batch_node_limit

    def __iter__(self):
        all_molecules, molecule_counts = self.molecule_idx.unique(return_counts=True)
        molecule_conformer_mapping = [
            list(range(sum(molecule_counts[:i]), sum(molecule_counts[:i + 1])))
            for i in range(len(all_molecules))]
        assert self.num_molecules == len(molecule_conformer_mapping)

        index = torch.randperm(self.num_molecules) if self.shuffle else torch.arange(self.num_molecules)

        batch_index = []
        current_num_nodes = 0
        for i in index:
            this_num_nodes = sum([self.num_nodes[i] for i in molecule_conformer_mapping[i]])
            if current_num_nodes > 0 and this_num_nodes >= self.node_limit:
                yield batch_index
                batch_index = []

            if self.strategy == 'all':
                batch_index += molecule_conformer_mapping[i]
            elif self.strategy == 'random':
                batch_index.append(random.choice(molecule_conformer_mapping[i]))
            elif self.strategy == 'first':
                batch_index.append(molecule_conformer_mapping[i][0])

            current_num_nodes += this_num_nodes
            if current_num_nodes > self.batch_node_limit or len(batch_index) >= self.batch_size:
                # print(current_num_nodes, [self.num_nodes[i].num_nodes for i in molecule_conformer_mapping[i]])
                yield batch_index
                batch_index = []
                current_num_nodes = 0
        if len(batch_index) > 0:
            yield batch_index


class EnsembleMultiBatchSampler:
    def __init__(self, dataset, batch_size, strategy='all', shuffle=True):
        assert strategy in ['all', 'random', 'first']
        self.molecule_idx = dataset.data.molecule_idx[dataset._indices]
        self.num_molecules = dataset.num_molecules
        self.batch_size = batch_size
        self.strategy = strategy
        self.shuffle = shuffle
        self.parts = dataset.data['part_id'][dataset._indices]

    def __iter__(self):
        all_molecules, molecule_counts = self.molecule_idx.unique(return_counts=True)
        cursor = 0
        molecule_conformer_mapping = []
        molecule_part_mapping = []
        for molecule, count in zip(all_molecules, molecule_counts):
            molecule_conformer_mapping.append(list(range(cursor, cursor + count)))
            molecule_part_mapping.append(self.parts[cursor:cursor + count])
            cursor += count
        assert self.num_molecules == len(molecule_conformer_mapping)

        index = torch.randperm(self.num_molecules) if self.shuffle else torch.arange(self.num_molecules)

        part_ids = torch.unique(self.parts).tolist()
        batch_indices = {p: [] for p in part_ids}

        for i in index:
            conformer_indices = molecule_conformer_mapping[i]
            parts = molecule_part_mapping[i]
            for current_part in part_ids:
                conformer_ids_by_var = [idx for idx, id_part in zip(conformer_indices, parts) if id_part == current_part]

                if self.strategy == 'all':
                    batch_indices[current_part].extend(conformer_ids_by_var)
                elif self.strategy == 'random':
                    id_var = random.choice(conformer_ids_by_var)
                    batch_indices[current_part].append(id_var)
                elif self.strategy == 'first':
                    batch_indices[current_part].append(conformer_ids_by_var[0])

            if len(batch_indices[part_ids[0]]) >= self.batch_size:
                yield [batch_indices[p] for p in part_ids]
                batch_indices = {p: [] for p in part_ids}
        if len(batch_indices[part_ids[0]]) > 0:
            yield [batch_indices[p] for p in part_ids]


class EnsembleBatchSamplerWithGraphLimit(Sampler):
    def __init__(self, dataset, batch_size=None, batch_graph_size=None, indices=None, shuffle=True):
        self.batch_size = batch_size if batch_size is not None else float('inf')
        self.batch_graph_size = batch_graph_size if batch_graph_size is not None else float('inf')
        self.shuffle = shuffle
        self.indices = [
            (idx, len(dataset.dataset.molecule_lists[0][data_idx]))
            for idx, data_idx in enumerate(dataset.indices)]
        # If indices are passed, then use only the ones passed (for DDP)
        if indices is not None:
            self.indices = torch.Tensor(self.indices)[indices].long().tolist()

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.indices)

        # self.pooled_indices = sorted(self.indices, key=lambda x: x[1], reverse=True)
        self.pooled_indices = self.indices
        batches = []
        current_batch = []
        current_length = 0

        for idx, length in self.pooled_indices:
            if len(current_batch) < self.batch_size and current_length + length <= self.batch_graph_size:
                current_batch.append(idx)
                current_length += length
            else:
                if len(current_batch) > 0:
                    batches.append(current_batch)
                current_batch = [idx]
                current_length = length

        if len(current_batch) > 0:
            batches.append(current_batch)

        print(len(batches), batches[0])
        for batch in batches:
            yield batch

    def __len__(self):
        return len(self.pooled_indices) // self.batch_size


class DistributedEnsembleSampler(DistributedSampler):
    def __init__(
            self, dataset, num_replicas: Optional[int] = None,
            rank: Optional[int] = None, shuffle: bool = True,
            seed: int = 0, drop_last: bool = False,
            batch_size: int = None, batch_graph_size: int = None) -> None:
        super().__init__(
            dataset=dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle, seed=seed, drop_last=drop_last)
        self.batch_size = batch_size
        self.batch_graph_size = batch_graph_size

    def __iter__(self):
        indices = list(super().__iter__())
        batch_sampler = EnsembleBatchSamplerWithGraphLimit(
            self.dataset, batch_size=self.batch_size, indices=indices, batch_graph_size=self.batch_graph_size)
        return iter(batch_sampler)

        # batch_sampler = EnsembleBatchSamplerWithGraphLimit(
        #     self.dataset, batch_size=self.batch_size, indices=indices, batch_graph_size=self.batch_graph_size)
        # batches = list(batch_sampler)
        #
        # # Gather batches from all replicas
        # gathered_batches = [None] * self.num_replicas
        # torch.distributed.all_gather_object(gathered_batches, batches)
        # print(gathered_batches)
        # TODO: should use dist.all_gather(gathered_batches, batches)
        #
        # # Flatten the gathered batches and select the ones corresponding to the current rank
        # flattened_batches = [batch for replica_batches in gathered_batches for batch in replica_batches]
        # num_batches_per_replica = len(flattened_batches) // self.num_replicas
        # start_idx = self.rank * num_batches_per_replica
        # end_idx = (self.rank + 1) * num_batches_per_replica
        # selected_batches = flattened_batches[start_idx:end_idx]
        #
        # return iter(selected_batches)

    def __len__(self) -> int:
        return self.num_samples // self.batch_size


if __name__ == '__main__':
    dataset = Drugs(root='../datasets/Drugs').shuffle()

    split = dataset.get_idx_split(train_ratio=0.1, valid_ratio=0.1)
    train_dataset = dataset[split['train']]
    valid_dataset = dataset[split['valid']]
    test_dataset = dataset[split['test']]
    test_loader = DataLoader(test_dataset, batch_sampler=EnsembleSampler(
        test_dataset, batch_size=32, shuffle=False))
    valid_loader = DataLoader(valid_dataset, batch_sampler=EnsembleSampler(
        valid_dataset, batch_size=32, shuffle=False))
    train_loader = DataLoader(train_dataset, batch_sampler=EnsembleSampler(
        train_dataset, batch_size=32, shuffle=True))

    dictionaries = {}
    with Chem.SDMolSupplier('../datasets/Drugs/raw/Drugs.sdf', removeHs=False) as suppl:
        import pdb; pdb.set_trace()
        for mol in suppl:
            id_ = mol.GetProp('ID')
            pos = mol.GetConformer().GetPositions()
            y = []
            for quantity in ['energy', 'ip', 'ea', 'chi', 'eta', 'omega']:
                y.append(float(mol.GetProp(quantity)))
            dictionaries[id_] = (pos, torch.Tensor(y))

    all_ids = set()
    all_names = set()
    conformer_labels = defaultdict(list)
    molecule_labels = {}

    for data in train_loader:
        unique_molecule_idx = torch.unique_consecutive(data.molecule_idx)
        for name, y in zip(data.name, data.y):
            conformer_labels[name].append(y)
        for id_ in unique_molecule_idx:
            name = data[data.molecule_idx == id_][0].name
            molecule_labels[name] = train_dataset.y[id_]

        for name in molecule_labels.keys():
            energy_list = [conformer[0].item() for conformer in conformer_labels[name]]
            for idx in range(6):
                quantity_list = [conformer[idx].item() for conformer in conformer_labels[name]]
                boltzmann_avg_quantity = boltzmann_average(quantity_list, energy_list)
                assert boltzmann_avg_quantity - molecule_labels[name][idx].item() < 1e-5

    for data in chain(train_loader, valid_loader, test_loader):
        all_ids |= set(data.id)
        all_names |= set(data.name)

        id_ = data.id[0]
        comp = data.pos[:2] == torch.from_numpy(dictionaries[id_][0])[:2]
        assert comp.all()

        comp = abs(data.y[0] - dictionaries[id_][1]) < 1e-5
        assert comp.all()

    assert len(all_ids) == len(dataset)
    assert len(all_names) == dataset.num_molecules
