from dataclasses import dataclass, field

from configs.model_1d import Model1D
from configs.model_2d import Model2D
from configs.model_3d import Model3D
from configs.model_4d import Model4D
from configs.model_dss import ModelDSS
from configs.model_fp_rf import ModelFPRF


@dataclass
class AdamW:
    betas: tuple = (0.9, 0.999)


@dataclass
class ReduceLROnPlateau:
    mode: str = 'min'
    factor: int = 0.8
    patience: int = 20


@dataclass
class CosineAnnealingLR:
    eta_min: float = 1e-6


@dataclass
class LinearWarmupCosineAnnealingLR:
    warmup_steps: int = 200
    max_epochs: int = 2000


@dataclass
class OneCycleLR:
    max_lr: float = 0.001
    steps_per_epoch: int = 100


@dataclass
class LambdaLR:
    lambda_type: str = 'cosine'
    warmup_factor: float = 0.2
    warmup_epochs: int = 5
    lr_min_factor: float = 1e-2


@dataclass
class Config:
    wandb_project: str = 'Auto4D'
    additional_notes: str = ''

    gpus: str = '0,1,2,3,4,5,6,7'
    port: str = None
    dataset: str = 'Kraken'
    target: str = 'qpoletens_xx'
    checkpoint: str = None
    max_num_molecules: int = None
    max_num_conformers: int = 20
    train_ratio: float = 0.7
    valid_ratio: float = 0.1

    batch_size: int = 256
    batch_graph_size: int = None
    hidden_dim: int = 128
    num_epochs: int = 500
    alert_epochs: int = 500
    patience: int = 200
    activation: str = 'relu'
    seed: int = 123
    device: str = 'cuda:5'
    dropout: float = 0.5

    optimizer: str = 'Adam'
    adamw: AdamW = field(default_factory=AdamW)
    scheduler: str = None
    reduce_lr_on_plateau: ReduceLROnPlateau = field(default_factory=ReduceLROnPlateau)
    cosine_annealing_lr: CosineAnnealingLR = field(default_factory=CosineAnnealingLR)
    linear_warmup_cosine_annealing_lr: LinearWarmupCosineAnnealingLR =\
        field(default_factory=LinearWarmupCosineAnnealingLR)
    one_cycle_lr: OneCycleLR = field(default_factory=OneCycleLR)
    lambda_lr: LambdaLR = field(default_factory=LambdaLR)

    learning_rate: float = 0.001
    weight_decay: float = 1e-4

    modelfprf: ModelFPRF = field(default_factory=ModelFPRF)
    model1d: Model1D = field(default_factory=Model1D)
    model2d: Model2D = field(default_factory=Model2D)
    model3d: Model3D = field(default_factory=Model3D)
    model4d: Model4D = field(default_factory=Model4D)
    modeldss: ModelDSS = field(default_factory=ModelDSS)
