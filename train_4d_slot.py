import os
import time
import uuid
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp

from dataclasses import asdict
from torch_geometric import seed_everything
from torch.utils.data import random_split
from torch.nn.parallel import DistributedDataParallel
from torch_geometric.loader import DataLoader

from config import Config
from data.kraken import KrakenV2 as Kraken
from happy_config import ConfigLoader
from loaders.multipart import MultiPartLoaderV2
from loaders.samplers import DistributedEnsembleSampler
from utils.checkpoint import load_checkpoint
from utils.early_stopping import EarlyStopping

from utils.optim import get_optimizer, get_scheduler

from models.model_4d_slot import SlotGATTest2


def train(model, loader, optimizer, rank):
    model.train()

    total_loss = torch.zeros(2).to(rank)
    for data in loader:
        optimizer.zero_grad()
        num_molecules = data[0].y.size(0)
        out = model(data)

        loss = F.l1_loss(out, data[0].y)
        loss.backward()
        optimizer.step()

        total_loss[0] += float(loss) * num_molecules
        total_loss[1] += num_molecules

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    loss = float(total_loss[0] / total_loss[1])
    return loss


def evaluate(model, loader, std):
    model.eval()
    error = 0
    num_molecules = 0

    for data in loader:
        with torch.no_grad():
            out = model.module(data)

        error += ((out - data[0].y) * std).abs().sum().item()
        num_molecules += data[0].y.size(0)
    return error / num_molecules

def run(rank, world_size, config):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = config.port
    dist.init_process_group('nccl', rank=rank, world_size=world_size)

    dataset = Kraken('datasets/Kraken', max_num_conformers=20)

    seed_everything(142)
    model = SlotGATTest2(
        num_parts=dataset.num_parts,
        edge_feat=False,
        num_edge_types=None,
        in_dim=128,
        hidden_dim=128,
        num_classes=64,
        num_layers=2,
        heads=[1, 3, 1],
        device=f'cuda:{rank}',
        feat_drop=0.1,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        activation=None,
        alpha=0.,
        max_atomic_num=54,
        max_num_confs=dataset.max_num_conformers
    ).to(rank)
    model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    target_id = dataset.descriptors.index(config.target)
    dataset.y = dataset.y[:, target_id]

    mean = dataset.y.mean(dim=0, keepdim=True)
    std = dataset.y.std(dim=0, keepdim=True)
    dataset.y = ((dataset.y - mean) / std).to(rank)
    mean = mean.to(rank)
    std = std.to(rank)

    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [config.train_ratio, config.valid_ratio, 1 - config.train_ratio - config.valid_ratio])

    if dataset.num_parts > 1:
        train_loader = MultiPartLoaderV2(
            train_dataset,
            num_parts=dataset.num_parts,
            batch_sampler=DistributedEnsembleSampler(
                dataset=train_dataset, num_replicas=world_size, rank=rank,
                batch_size=config.batch_size, batch_graph_size=config.batch_graph_size))
        if rank == 0:
            valid_loader = MultiPartLoaderV2(valid_dataset, num_parts=dataset.num_parts, batch_size=8)
            test_loader = MultiPartLoaderV2(test_dataset, num_parts=dataset.num_parts, batch_size=8)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=DistributedEnsembleSampler(
                dataset=train_dataset, num_replicas=world_size, rank=rank,
                batch_size=config.batch_size, batch_graph_size=config.batch_graph_size))

        if rank == 0:
            valid_loader = DataLoader(valid_dataset, batch_size=8)
            test_loader = DataLoader(test_dataset, batch_size=8)

    optimizer = get_optimizer(model.parameters(), config)
    scheduler = get_scheduler(optimizer, config, train_dataset, world_size)

    start_epoch = 0
    checkpoint_path = (
        f'checkpoints/'
        f'{config.dataset}_{config.target}_'
        f'{uuid.uuid4()}.pt')
    if rank == 0:
        print(f'Saving checkpoint to: {checkpoint_path}')
    if os.path.exists(checkpoint_path):
        model, optimizer, start_epoch = load_checkpoint(checkpoint_path, model, optimizer)
        print(f'Loaded checkpoint: {checkpoint_path} at epoch {start_epoch} on rank {rank}')
        dist.barrier()
    #--if rank == 0:
    if rank == 0:
        early_stopping = EarlyStopping(patience=config.patience, path=checkpoint_path)
        #wandb.init(project='Auto4D', config=asdict(config))
        #wandb.define_metric('epoch')
        #wandb.define_metric('train_loss', step_metric='epoch')
        #wandb.define_metric('valid_error', step_metric='epoch')
        #wandb.define_metric('test_error', step_metric='epoch')
    else:
        early_stopping = None

    for epoch in range(start_epoch, config.num_epochs):
        train_loader.batch_sampler.set_epoch(epoch)
        loss = train(model, train_loader, optimizer, rank)
        if scheduler is not None:
            scheduler.step(loss)
        print(f'Rank: {rank}, Epoch: {epoch}/{config.num_epochs}, Loss: {loss:.5f}')
        if rank == 0:
            valid_error = evaluate(model, valid_loader, std)
            early_stopping(valid_error, model, optimizer, epoch)
            if early_stopping.counter == 0:
                test_error = evaluate(model, test_loader, std)
            if early_stopping.early_stop:
                print('Early stopping...')
                break
            print(f'Progress: {epoch}/{config.num_epochs}/{loss:.5f}/{valid_error:.5f}/{test_error:.5f}')

            #if epoch % config.alert_epochs == 0:
            #    wandb.alert(
            #        title=f'{epoch} epochs reached',
            #        text=f'{epoch} epochs reached on '
            #             f'{config.dataset} ({config.target}) using '
            #             f'SlotGAT')
        dist.barrier()

        if early_stopping is not None:
            early_stop = torch.tensor([early_stopping.early_stop], device=rank)
        else:
            early_stop = torch.tensor([False], device=rank)
        dist.broadcast(early_stop, src=0)
        if early_stop.item():
            break
    #if rank == 9924012431840921840: # 0
        #wandb.finish()
    dist.destroy_process_group()


# TODO: Testing function
def test_model(rank, config):
    dataset = Kraken('datasets/Kraken', max_num_conformers=3)

    # max_atomic_num = 100

    seed_everything(142)
    model = SlotGATTest2(
        num_parts=dataset.num_parts,
        edge_feat=False,
        num_edge_types=None,
        in_dim=8,
        hidden_dim=16,
        num_classes=16,
        num_layers=2,
        heads=[1, 1, 1, 1],
        device=f'cuda:{rank}',
        feat_drop=0.1,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        activation=None,
        alpha=0.,
        max_atomic_num=16,
        max_num_confs=dataset.max_num_conformers
    ).to(rank)
    # model = DistributedDataParallel(model, device_ids=[rank], find_unused_parameters=True)

    target_id = dataset.descriptors.index(config.target)
    dataset.y = dataset.y[:, target_id]

    dataset = dataset[:50]
    sample_loader = DataLoader(
        dataset,
        batch_size=8)

    model.eval()
    out = 0
    for data in sample_loader:
        print("--------------------PASSING THROUGH MODEL ---------------------")
        out = model(data)
        break
    return out

if __name__ == '__main__':
    loader = ConfigLoader(model=Config, config='params/params_slot.json')
    config = loader()

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')

    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus

    world_size = torch.cuda.device_count()
    print(f"Let's use {world_size} GPUs!")
    time_start = time.time()
    args = (world_size, config)
    mp.spawn(run, args=args, nprocs=world_size, join=True)
    #test_model(rank=0, config=config)
    time_end = time.time()
    print(f'Total time: {time_end - time_start:.2f} seconds')
