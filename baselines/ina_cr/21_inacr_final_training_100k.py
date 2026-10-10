"""INA-CR final clean training for the geospatial baseline.

Frozen protocol
---------------
Training/reference population:
    first 100,000 rows of the authoritative seed-7 300,000-row baseline pool

Views:
    terrain [5], hydrological [3], spatial [4], surface [3],
    infrastructure [1], landuse [7]

Model/adaptation:
    one view-specific encoder and decoder per view
    one prediction module per view
    shared latent width = 32
    encoder: Linear(D_v, 32) + Tanh
    decoder: Linear(32, D_v) + Sigmoid
    prediction: Linear(32, 32) + Tanh

Training:
    k=6 global neighbors in concatenated 23-D [0,1] clean-reference space
    batch size=100
    learning rate=1e-4
    Adam AMSGrad, weight_decay=1e-6
    model seed=1 (matching released INA-CR code)
    200 epochs PER VIEW, one round
    weights (reconstruction, neighborhood, prediction, instance)
            = (1.0, 0.25, 1.0, 0.25)

Preprocessing:
    existing project robust preprocessing
    then INA-CR-specific feature-wise clean-reference min-max to [0,1], clipped

Post-training frozen scoring artifacts:
    clean-reference latent means
    clean-reference per-view/per-component score min/max ranges
    no clipping of score extrapolation after component normalization

No injected data, anomaly labels, AUC, F1, or detection-rate values are read or used.

The script checkpoints after EVERY epoch. If interrupted, rerun with --resume.
"""
from __future__ import annotations

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Frozen final protocol
# =============================================================================

DEFAULT_ROOT = Path(os.environ.get("GIS_ANOMALY_PROJECT_ROOT", Path.cwd())).resolve()

EXPECTED_N = 4_290_958
EXPECTED_D = 23
BASELINE_FIT_SIZE = 300_000
FINAL_TRAIN_N = 100_000
EXPECTED_BASELINE_SUBSET_SEED = 7

MODEL_SEED = 1
LATENT_DIM = 32
BATCH_SIZE = 100
LEARNING_RATE = 1e-4
K_NEIBS = 6
NUM_EPOCHS = 200
NUM_ROUNDS = 1
WEIGHT_DECAY = 1e-6
WEIGHTS = (1.0, 0.25, 1.0, 0.25)  # rec, neighborhood, prediction, instance

VIEW_ORDER = [
    "terrain",
    "hydrological",
    "spatial",
    "surface",
    "infrastructure",
    "landuse",
]

VIEW_DIMS = {
    "terrain": 5,
    "hydrological": 3,
    "spatial": 4,
    "surface": 3,
    "infrastructure": 1,
    "landuse": 7,
}

VIEW_INDICES = {
    "terrain": [0, 1, 3, 4, 5],
    "hydrological": [2, 10, 12],
    "spatial": [6, 7, 8, 9],
    "surface": [13, 14, 15],
    "infrastructure": [11],
    "landuse": list(range(16, 23)),
}

COMPONENTS = ["reconstruction", "neighborhood", "prediction", "instance"]


# =============================================================================
# Paths / basic helpers
# =============================================================================

def require_file(path: Path, label: str = "") -> Path:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {label or 'required file'}:\n{path}"
        )
    return path


def resolve_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root).expanduser().resolve()

    here = Path(__file__).resolve().parent

    if (here / "X.npy").is_file():
        return here

    if (here.parent / "X.npy").is_file():
        return here.parent

    return DEFAULT_ROOT


def atomic_torch_save(obj, path: Path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def write_json(path: Path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
    tmp.replace(path)


# =============================================================================
# Existing project preprocessing
# =============================================================================

@dataclass
class ViewPreprocessor:
    terrain_scaler: object
    hydro_scaler: object
    spatial_scaler: object
    surface_scaler: object
    infra_scaler: object
    terrain_bounds: dict
    spatial_bounds: dict

    @classmethod
    def load(cls, clean_dir: Path) -> "ViewPreprocessor":
        files = {
            "terrain_scaler": clean_dir / "terrain_scaler.pkl",
            "hydro_scaler": clean_dir / "hydro_scaler.pkl",
            "spatial_scaler": clean_dir / "spatial_scaler.pkl",
            "surface_scaler": clean_dir / "surface_scaler.pkl",
            "infra_scaler": clean_dir / "infra_scaler.pkl",
            "terrain_bounds": clean_dir / "terrain_clip_bounds.pkl",
            "spatial_bounds": clean_dir / "spatial_clip_bounds.pkl",
        }

        for name, path in files.items():
            require_file(path, name)

        return cls(
            terrain_scaler=joblib.load(files["terrain_scaler"]),
            hydro_scaler=joblib.load(files["hydro_scaler"]),
            spatial_scaler=joblib.load(files["spatial_scaler"]),
            surface_scaler=joblib.load(files["surface_scaler"]),
            infra_scaler=joblib.load(files["infra_scaler"]),
            terrain_bounds=joblib.load(files["terrain_bounds"]),
            spatial_bounds=joblib.load(files["spatial_bounds"]),
        )

    def transform(self, raw: np.ndarray) -> Dict[str, np.ndarray]:
        raw = np.asarray(raw)

        if (
            raw.ndim != 2
            or raw.shape[1] != EXPECTED_D
            or not np.isfinite(raw).all()
        ):
            raise ValueError(f"Invalid raw feature block {raw.shape}")

        terrain = raw[:, VIEW_INDICES["terrain"]].astype(
            np.float32, copy=True
        )

        for fname, pos in {
            "profile_curv": 2,
            "plan_curv": 3,
        }.items():
            b = self.terrain_bounds[fname]
            terrain[:, pos] = np.tanh(
                np.clip(
                    terrain[:, pos],
                    float(b["lower"]),
                    float(b["upper"]),
                )
            )

        terrain[:, 4] = np.log1p(
            np.clip(terrain[:, 4], 0.0, None)
        )

        terrain = self.terrain_scaler.transform(
            terrain
        ).astype(np.float32, copy=False)

        hydro = self.hydro_scaler.transform(
            raw[:, VIEW_INDICES["hydrological"]]
        ).astype(np.float32, copy=False)

        spatial = raw[:, VIEW_INDICES["spatial"]].astype(
            np.float32, copy=True
        )

        for pos, fname in enumerate(
            [
                "local_std",
                "elev_percentile",
                "elev_contrast",
                "slope_variability",
            ]
        ):
            b = self.spatial_bounds[fname]
            spatial[:, pos] = np.clip(
                spatial[:, pos],
                float(b["lower"]),
                float(b["upper"]),
            )

        spatial = self.spatial_scaler.transform(
            spatial
        ).astype(np.float32, copy=False)

        surface = self.surface_scaler.transform(
            raw[:, VIEW_INDICES["surface"]]
        ).astype(np.float32, copy=False)

        infra = self.infra_scaler.transform(
            raw[:, VIEW_INDICES["infrastructure"]]
        ).astype(np.float32, copy=False)

        landuse = raw[:, VIEW_INDICES["landuse"]].astype(
            np.float32, copy=True
        )

        views = {
            "terrain": terrain,
            "hydrological": hydro,
            "spatial": spatial,
            "surface": surface,
            "infrastructure": infra,
            "landuse": landuse,
        }

        n = raw.shape[0]

        for name in VIEW_ORDER:
            if (
                views[name].shape != (n, VIEW_DIMS[name])
                or not np.isfinite(views[name]).all()
            ):
                raise ValueError(
                    f"Invalid processed {name} view "
                    f"{views[name].shape}"
                )

        return views


# =============================================================================
# INA-CR-specific clean-reference [0,1] scaling
# =============================================================================

@dataclass
class MinMax01:
    mins: Dict[str, np.ndarray]
    maxs: Dict[str, np.ndarray]

    @classmethod
    def fit(
        cls,
        views: Dict[str, np.ndarray],
    ) -> "MinMax01":
        return cls(
            mins={
                name: np.min(
                    views[name],
                    axis=0,
                ).astype(np.float32)
                for name in VIEW_ORDER
            },
            maxs={
                name: np.max(
                    views[name],
                    axis=0,
                ).astype(np.float32)
                for name in VIEW_ORDER
            },
        )

    def transform(
        self,
        views: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        out = {}

        for name in VIEW_ORDER:
            arr = np.asarray(
                views[name],
                dtype=np.float32,
            )

            lo = self.mins[name]
            hi = self.maxs[name]
            den = hi - lo

            safe = np.where(
                den > 0,
                den,
                1.0,
            ).astype(np.float32)

            z = (arr - lo) / safe

            if np.any(den == 0):
                z[:, den == 0] = 0.0

            out[name] = np.clip(
                z,
                0.0,
                1.0,
            ).astype(np.float32, copy=False)

        return out


# =============================================================================
# Dataset
# =============================================================================

class GeoViewDataset(Dataset):
    def __init__(
        self,
        view_data: np.ndarray,
        global_neighbors: np.ndarray,
    ):
        self.data = torch.from_numpy(
            np.asarray(
                view_data,
                dtype=np.float32,
            )
        )

        self.global_neighbors = torch.from_numpy(
            np.asarray(
                global_neighbors,
                dtype=np.int64,
            )
        )

    def __len__(self):
        return int(self.data.shape[0])

    def __getitem__(self, index):
        ng = self.global_neighbors[index]

        return (
            self.data[index],
            int(index),
            self.data[ng],
        )


# =============================================================================
# Generalized INA-CR networks
# =============================================================================

class GeneratorEncode(nn.Module):
    def __init__(
        self,
        data_dim: int,
        rep_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self.rep_dim = int(rep_dim)

        self.encoder = nn.Sequential(
            nn.Linear(
                int(data_dim),
                self.rep_dim,
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


class GeneratorDecode(nn.Module):
    def __init__(
        self,
        data_dim: int,
        rep_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(
                int(rep_dim),
                int(data_dim),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(x)


class Prediction(nn.Module):
    def __init__(
        self,
        rep_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(
                int(rep_dim),
                int(rep_dim),
            ),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.encoder(x)


# =============================================================================
# Model-state helpers
# =============================================================================

def model_state(
    encoders,
    decoders,
    predictors,
):
    return {
        "encoders": [
            m.state_dict()
            for m in encoders
        ],
        "decoders": [
            m.state_dict()
            for m in decoders
        ],
        "predictors": [
            m.state_dict()
            for m in predictors
        ],
    }


def load_model_state(
    state,
    encoders,
    decoders,
    predictors,
):
    for model, sd in zip(
        encoders,
        state["encoders"],
    ):
        model.load_state_dict(sd)

    for model, sd in zip(
        decoders,
        state["decoders"],
    ):
        model.load_state_dict(sd)

    for model, sd in zip(
        predictors,
        state["predictors"],
    ):
        model.load_state_dict(sd)


# =============================================================================
# Training
# =============================================================================

class Trainer:
    def __init__(
        self,
        device: str,
        out_dir: Path,
    ):
        self.device = device
        self.out_dir = out_dir
        self.checkpoint_path = (
            out_dir / "INA_CR_training_checkpoint.pt"
        )
        self.loss_log_path = (
            out_dir / "INA_CR_training_loss_log.csv"
        )

    @staticmethod
    def _batch_views(
        view_tensors: List[torch.Tensor],
        index: torch.Tensor,
        device: str,
    ) -> List[torch.Tensor]:
        idx = index.long()

        return [
            v[idx].to(
                device,
                non_blocking=True,
            )
            for v in view_tensors
        ]

    def _append_loss(
        self,
        view_id,
        view_name,
        epoch,
        elapsed,
        loss,
    ):
        new_file = not self.loss_log_path.exists()

        with self.loss_log_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "view_id",
                    "view_name",
                    "epoch",
                    "epoch_seconds",
                    "mean_gloss",
                ],
            )

            if new_file:
                writer.writeheader()

            writer.writerow(
                {
                    "view_id": int(view_id),
                    "view_name": view_name,
                    "epoch": int(epoch),
                    "epoch_seconds": float(elapsed),
                    "mean_gloss": float(loss),
                }
            )

    def train_all(
        self,
        datasets,
        view_tensors,
        encoders,
        decoders,
        predictors,
        resume: bool,
    ):
        start_view = 0
        start_epoch = 0
        resume_payload = None

        if self.checkpoint_path.exists():
            if not resume:
                raise RuntimeError(
                    "\nA training checkpoint already exists:\n"
                    f"{self.checkpoint_path}\n\n"
                    "If this is an interrupted run, rerun with:\n"
                    "  python 21_inacr_final_training_100k.py --resume\n"
                )

            resume_payload = torch.load(
                self.checkpoint_path,
                map_location=self.device,
                weights_only=False,
            )

            load_model_state(
                resume_payload["model_state"],
                encoders,
                decoders,
                predictors,
            )

            start_view = int(
                resume_payload["current_view"]
            )
            start_epoch = int(
                resume_payload["next_epoch"]
            )

            torch.set_rng_state(
                resume_payload["torch_rng_state"]
            )

            if (
                self.device == "cuda"
                and resume_payload.get(
                    "cuda_rng_state_all"
                ) is not None
            ):
                torch.cuda.set_rng_state_all(
                    resume_payload[
                        "cuda_rng_state_all"
                    ]
                )

            np.random.set_state(
                resume_payload["numpy_rng_state"]
            )

            print(
                f"Resuming from view "
                f"{start_view + 1}/{len(VIEW_ORDER)}, "
                f"epoch {start_epoch + 1}/{NUM_EPOCHS}"
            )

        elif resume:
            raise RuntimeError(
                "--resume was requested but no checkpoint exists."
            )

        for id_view, name in enumerate(VIEW_ORDER):
            if id_view < start_view:
                print(
                    f"Skipping completed view "
                    f"{id_view + 1}: {name}"
                )
                continue

            print("\n" + "-" * 80)
            print(
                f"FINAL TRAINING VIEW "
                f"{id_view + 1}/{len(VIEW_ORDER)}: "
                f"{name} (D={VIEW_DIMS[name]})"
            )
            print("-" * 80)

            for m in encoders:
                m.to(self.device)
                m.train()

            decoders[id_view].to(self.device)
            decoders[id_view].train()

            predictors[id_view].to(self.device)
            predictors[id_view].train()

            enc_opts = [
                optim.Adam(
                    m.parameters(),
                    lr=LEARNING_RATE,
                    weight_decay=WEIGHT_DECAY,
                    amsgrad=True,
                )
                for m in encoders
            ]

            dec_opt = optim.Adam(
                decoders[id_view].parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                amsgrad=True,
            )

            pred_opt = optim.Adam(
                predictors[id_view].parameters(),
                lr=LEARNING_RATE,
                weight_decay=WEIGHT_DECAY,
                amsgrad=True,
            )

            shuffle_gen = torch.Generator()

            epoch0 = 0

            if (
                resume_payload is not None
                and id_view == start_view
                and start_epoch > 0
            ):
                for opt, sd in zip(
                    enc_opts,
                    resume_payload[
                        "encoder_optimizer_states"
                    ],
                ):
                    opt.load_state_dict(sd)

                dec_opt.load_state_dict(
                    resume_payload[
                        "decoder_optimizer_state"
                    ]
                )

                pred_opt.load_state_dict(
                    resume_payload[
                        "prediction_optimizer_state"
                    ]
                )

                shuffle_gen.set_state(
                    resume_payload[
                        "dataloader_generator_state"
                    ]
                )

                epoch0 = start_epoch

            else:
                # Deterministic view-specific shuffle sequence.
                shuffle_gen.manual_seed(
                    MODEL_SEED + 10_000 + id_view
                )
                epoch0 = 0

            loader = DataLoader(
                datasets[id_view],
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                drop_last=False,
                generator=shuffle_gen,
            )

            for epoch in range(
                epoch0,
                NUM_EPOCHS,
            ):
                t0 = time.time()
                loss_sum = 0.0
                nb = 0

                for (
                    inputs,
                    index,
                    neighbor_global,
                ) in loader:
                    inputs = inputs.to(
                        self.device,
                        non_blocking=True,
                    )

                    neighbor_global = (
                        neighbor_global.to(
                            self.device,
                            non_blocking=True,
                        )
                    )

                    batch_views = (
                        self._batch_views(
                            view_tensors,
                            index,
                            self.device,
                        )
                    )

                    for op in enc_opts:
                        op.zero_grad(
                            set_to_none=True
                        )

                    dec_opt.zero_grad(
                        set_to_none=True
                    )

                    pred_opt.zero_grad(
                        set_to_none=True
                    )

                    encoded = encoders[
                        id_view
                    ](inputs)

                    enc_ng = encoders[
                        id_view
                    ](
                        neighbor_global
                    ).mean(dim=1)

                    outputs = decoders[
                        id_view
                    ](encoded)

                    # Preserve released training-code
                    # normalization exactly.
                    recon_error = (
                        torch.sum(
                            (
                                outputs
                                - inputs
                            )
                            ** 2,
                            dim=tuple(
                                range(
                                    1,
                                    outputs.dim(),
                                )
                            ),
                        )
                        / len(inputs)
                    )

                    all_latents = torch.stack(
                        [
                            encoders[i](
                                batch_views[i]
                            )
                            for i in range(
                                len(encoders)
                            )
                        ],
                        dim=0,
                    )

                    mean_latent = (
                        all_latents.mean(dim=0)
                    )

                    prediction_loss = (
                        torch.sum(
                            (
                                mean_latent
                                - predictors[
                                    id_view
                                ](encoded)
                            )
                            ** 2,
                            dim=1,
                        )
                        / encoded.shape[1]
                    )

                    ins_sum = torch.zeros(
                        len(inputs),
                        device=self.device,
                    )

                    for latent_j in all_latents:
                        aaa = torch.exp(
                            F.cosine_similarity(
                                encoded,
                                latent_j,
                                dim=1,
                            )
                        )

                        bbb = (
                            len(
                                all_latents[
                                    id_view
                                ]
                            )
                            * torch.exp(
                                F.cosine_similarity(
                                    encoded,
                                    torch.mean(
                                        latent_j,
                                        dim=0,
                                    ),
                                    dim=1,
                                )
                            )
                        )

                        ins_sum += -torch.log(
                            aaa / bbb
                        )

                    ins_sum /= (
                        len(encoders)
                        * encoded.shape[1]
                    )

                    nei_sum = torch.zeros(
                        len(inputs),
                        device=self.device,
                    )

                    for latent_j in all_latents:
                        aaa = torch.exp(
                            F.cosine_similarity(
                                enc_ng,
                                latent_j,
                                dim=1,
                            )
                        )

                        bbb = (
                            len(
                                all_latents[
                                    id_view
                                ]
                            )
                            * torch.exp(
                                F.cosine_similarity(
                                    enc_ng,
                                    torch.mean(
                                        latent_j,
                                        dim=0,
                                    ),
                                    dim=1,
                                )
                            )
                            + torch.exp(
                                F.cosine_similarity(
                                    enc_ng,
                                    torch.mean(
                                        encoded,
                                        dim=0,
                                    ),
                                    dim=1,
                                )
                            )
                        )

                        nei_sum += -torch.log(
                            aaa / bbb
                        )

                    nei_sum /= (
                        len(encoders)
                        * encoded.shape[1]
                    )

                    scores = (
                        WEIGHTS[0]
                        * recon_error
                        + WEIGHTS[1]
                        * nei_sum
                        + WEIGHTS[2]
                        * prediction_loss
                        + WEIGHTS[3]
                        * ins_sum
                    )

                    loss = torch.mean(scores)

                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            "Non-finite training loss "
                            f"in view {id_view}, "
                            f"epoch {epoch + 1}"
                        )

                    loss.backward()

                    for op in enc_opts:
                        op.step()

                    dec_opt.step()
                    pred_opt.step()

                    loss_sum += float(
                        loss.item()
                    )
                    nb += 1

                elapsed = time.time() - t0
                mean_loss = (
                    loss_sum / max(nb, 1)
                )

                print(
                    f" View {id_view + 1} "
                    f"Round 0 "
                    f"Epoch {epoch + 1}/{NUM_EPOCHS}\t"
                    f"Time: {elapsed:.3f}\t"
                    f"GLoss: {mean_loss:.8f}"
                )

                self._append_loss(
                    id_view,
                    name,
                    epoch + 1,
                    elapsed,
                    mean_loss,
                )

                # Checkpoint AFTER every fully completed epoch.
                if epoch + 1 < NUM_EPOCHS:
                    next_view = id_view
                    next_epoch = epoch + 1

                    payload = {
                        "current_view": next_view,
                        "next_epoch": next_epoch,
                        "model_state": model_state(
                            encoders,
                            decoders,
                            predictors,
                        ),
                        "encoder_optimizer_states": [
                            op.state_dict()
                            for op in enc_opts
                        ],
                        "decoder_optimizer_state":
                            dec_opt.state_dict(),
                        "prediction_optimizer_state":
                            pred_opt.state_dict(),
                        "dataloader_generator_state":
                            shuffle_gen.get_state(),
                        "torch_rng_state":
                            torch.get_rng_state(),
                        "cuda_rng_state_all":
                            (
                                torch.cuda.get_rng_state_all()
                                if self.device == "cuda"
                                else None
                            ),
                        "numpy_rng_state":
                            np.random.get_state(),
                        "completed_epoch":
                            epoch + 1,
                        "view_name": name,
                    }

                else:
                    # Current view completed. The released
                    # implementation recreates optimizers for
                    # the next view, so optimizer state is not
                    # carried across views.
                    payload = {
                        "current_view":
                            id_view + 1,
                        "next_epoch": 0,
                        "model_state": model_state(
                            encoders,
                            decoders,
                            predictors,
                        ),
                        "encoder_optimizer_states": [],
                        "decoder_optimizer_state": None,
                        "prediction_optimizer_state": None,
                        "dataloader_generator_state":
                            shuffle_gen.get_state(),
                        "torch_rng_state":
                            torch.get_rng_state(),
                        "cuda_rng_state_all":
                            (
                                torch.cuda.get_rng_state_all()
                                if self.device == "cuda"
                                else None
                            ),
                        "numpy_rng_state":
                            np.random.get_state(),
                        "completed_epoch":
                            NUM_EPOCHS,
                        "view_name": name,
                    }

                atomic_torch_save(
                    payload,
                    self.checkpoint_path,
                )

            # From here on, this view is complete.
            start_epoch = 0
            resume_payload = None

        print(
            "\nAll six views completed "
            "200 epochs per view."
        )

        return encoders, decoders, predictors


# =============================================================================
# Frozen inductive scoring functions validated in Stage D2
# =============================================================================

def encode_all(
    encoders,
    views,
    device,
    chunk=4096,
):
    for m in encoders:
        m.eval()

    n = len(
        next(iter(views.values()))
    )

    out = [
        []
        for _ in encoders
    ]

    with torch.no_grad():
        for start in range(
            0,
            n,
            chunk,
        ):
            end = min(
                start + chunk,
                n,
            )

            for i, name in enumerate(
                VIEW_ORDER
            ):
                x = torch.from_numpy(
                    views[name][start:end]
                ).to(device)

                out[i].append(
                    encoders[i](x)
                    .cpu()
                    .numpy()
                )

    return [
        np.concatenate(
            parts,
            axis=0,
        ).astype(
            np.float32,
            copy=False,
        )
        for parts in out
    ]


def frozen_latent_means(
    encoders,
    ref_views,
    device,
):
    latent = encode_all(
        encoders,
        ref_views,
        device,
    )

    means = [
        arr.mean(
            axis=0
        ).astype(np.float32)
        for arr in latent
    ]

    return latent, means


def score_raw_inductive(
    query_views,
    query_neighbor_idx,
    ref_views,
    encoders,
    decoders,
    predictors,
    ref_latent_means,
    device,
    chunk_size=1024,
):
    """
    Query-independent scoring validated in Stage D2.

    The source-code test-batch latent means are replaced
    by frozen CLEAN-reference latent means.

    The released batch-size coefficient (=100) is retained
    in the alignment denominators.
    """
    n = query_neighbor_idx.shape[0]

    for m in encoders:
        m.eval()

    for m in decoders:
        m.eval()

    for m in predictors:
        m.eval()

    mean_t = [
        torch.from_numpy(
            mean
        ).to(device)
        for mean in ref_latent_means
    ]

    per_view = []

    with torch.no_grad():
        for v, name in enumerate(
            VIEW_ORDER
        ):
            rec_parts = []
            nei_parts = []
            pred_parts = []
            ins_parts = []

            for start in range(
                0,
                n,
                chunk_size,
            ):
                end = min(
                    start + chunk_size,
                    n,
                )

                batch_q = [
                    torch.from_numpy(
                        query_views[nm][
                            start:end
                        ]
                    ).to(device)
                    for nm in VIEW_ORDER
                ]

                encoded_all = [
                    encoders[j](
                        batch_q[j]
                    )
                    for j in range(
                        len(VIEW_ORDER)
                    )
                ]

                encoded = encoded_all[v]

                recon = torch.sum(
                    (
                        decoders[v](encoded)
                        - batch_q[v]
                    )
                    ** 2,
                    dim=1,
                )

                mean_query_lat = (
                    torch.stack(
                        encoded_all,
                        dim=0,
                    ).mean(dim=0)
                )

                pred = (
                    torch.sum(
                        (
                            mean_query_lat
                            - predictors[v](
                                encoded
                            )
                        )
                        ** 2,
                        dim=1,
                    )
                    / LATENT_DIM
                )

                ng_idx = (
                    query_neighbor_idx[
                        start:end
                    ]
                )

                ng_np = ref_views[
                    name
                ][ng_idx]

                ng = torch.from_numpy(
                    np.asarray(
                        ng_np,
                        dtype=np.float32,
                    )
                ).to(device)

                enc_ng = encoders[v](
                    ng
                ).mean(dim=1)

                ins = torch.zeros(
                    end - start,
                    device=device,
                )

                nei = torch.zeros(
                    end - start,
                    device=device,
                )

                for j, latent_j in enumerate(
                    encoded_all
                ):
                    aaa_i = torch.exp(
                        F.cosine_similarity(
                            encoded,
                            latent_j,
                            dim=1,
                        )
                    )

                    bbb_i = (
                        BATCH_SIZE
                        * torch.exp(
                            F.cosine_similarity(
                                encoded,
                                mean_t[j][
                                    None, :
                                ],
                                dim=1,
                            )
                        )
                    )

                    ins += -torch.log(
                        aaa_i / bbb_i
                    )

                    aaa_n = torch.exp(
                        F.cosine_similarity(
                            enc_ng,
                            latent_j,
                            dim=1,
                        )
                    )

                    bbb_n = (
                        BATCH_SIZE
                        * torch.exp(
                            F.cosine_similarity(
                                enc_ng,
                                mean_t[j][
                                    None, :
                                ],
                                dim=1,
                            )
                        )
                        + torch.exp(
                            F.cosine_similarity(
                                enc_ng,
                                mean_t[v][
                                    None, :
                                ],
                                dim=1,
                            )
                        )
                    )

                    nei += -torch.log(
                        aaa_n / bbb_n
                    )

                ins /= (
                    len(VIEW_ORDER)
                    * LATENT_DIM
                )

                nei /= (
                    len(VIEW_ORDER)
                    * LATENT_DIM
                )

                rec_parts.append(
                    recon.cpu().numpy()
                )

                nei_parts.append(
                    nei.cpu().numpy()
                )

                pred_parts.append(
                    pred.cpu().numpy()
                )

                ins_parts.append(
                    ins.cpu().numpy()
                )

            per_view.append(
                {
                    "reconstruction":
                        np.concatenate(
                            rec_parts
                        ),
                    "neighborhood":
                        np.concatenate(
                            nei_parts
                        ),
                    "prediction":
                        np.concatenate(
                            pred_parts
                        ),
                    "instance":
                        np.concatenate(
                            ins_parts
                        ),
                }
            )

    return per_view


def fit_score_ranges(
    per_view_raw,
):
    ranges = []

    for v in per_view_raw:
        rd = {}

        for comp in COMPONENTS:
            arr = np.asarray(
                v[comp],
                dtype=np.float64,
            )

            lo = float(np.min(arr))
            hi = float(np.max(arr))

            if not np.isfinite(
                [lo, hi]
            ).all():
                raise FloatingPointError(
                    "Non-finite score range"
                )

            rd[comp] = (
                lo,
                hi,
            )

        ranges.append(rd)

    return ranges


def aggregate_frozen(
    per_view_raw,
    ranges,
):
    n = len(
        per_view_raw[0][
            "reconstruction"
        ]
    )

    agg = {
        comp: np.zeros(
            n,
            dtype=np.float64,
        )
        for comp in COMPONENTS
    }

    for vi, view in enumerate(
        per_view_raw
    ):
        for comp in COMPONENTS:
            lo, hi = ranges[
                vi
            ][comp]

            arr = np.asarray(
                view[comp],
                dtype=np.float64,
            )

            if hi - lo == 0:
                z = arr.copy()
            else:
                # Intentionally no clipping:
                # preserve test-score extrapolation.
                z = (
                    arr - lo
                ) / (
                    hi - lo
                )

            agg[comp] += z

    total = (
        WEIGHTS[0]
        * agg["reconstruction"]
        + WEIGHTS[1]
        * agg["neighborhood"]
        + WEIGHTS[2]
        * agg["prediction"]
        + WEIGHTS[3]
        * agg["instance"]
    )

    return agg, total


# =============================================================================
# Final artifact saving
# =============================================================================

def save_input_ranges(
    out_dir,
    mm01,
):
    np.savez(
        out_dir
        / "INA_CR_input_minmax01_ranges.npz",
        **{
            f"{name}_min":
                mm01.mins[name]
            for name in VIEW_ORDER
        },
        **{
            f"{name}_max":
                mm01.maxs[name]
            for name in VIEW_ORDER
        },
    )


def save_component_ranges(
    out_dir,
    ranges,
):
    rows = []

    payload = {}

    for vi, name in enumerate(
        VIEW_ORDER
    ):
        row = {
            "view": name
        }

        for comp in COMPONENTS:
            lo, hi = ranges[
                vi
            ][comp]

            row[
                f"{comp}_clean_min"
            ] = lo

            row[
                f"{comp}_clean_max"
            ] = hi

            payload[
                f"{name}_{comp}_min"
            ] = np.array(
                lo,
                dtype=np.float64,
            )

            payload[
                f"{name}_{comp}_max"
            ] = np.array(
                hi,
                dtype=np.float64,
            )

        rows.append(row)

    with (
        out_dir
        / "INA_CR_frozen_component_ranges.csv"
    ).open(
        "w",
        newline="",
        encoding="utf-8",
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        out_dir
        / "INA_CR_frozen_component_ranges.npz",
        **payload,
    )


# =============================================================================
# Main
# =============================================================================

def main(
    cli_root: str | None = None,
    resume: bool = False,
):
    root = resolve_root(cli_root)

    clean_dir = (
        root
        / "03_Proposed_Results_Seed_42"
    )

    baseline_dir = (
        root
        / "02_Baseline_Results_Seed_42"
    )

    out_dir = (
        root
        / "21_INACR_Final_Training_100k"
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 80)
    print(
        "INA-CR FINAL CLEAN TRAINING "
        "— 100,000 REFERENCE PIXELS"
    )
    print(
        "No injected data or final-test "
        "labels are used."
    )
    print("=" * 80)

    print("Project root:", root)
    print("Output:", out_dir)

    torch.manual_seed(
        MODEL_SEED
    )
    np.random.seed(
        MODEL_SEED
    )

    cuda_ok = (
        torch.cuda.is_available()
    )

    device = (
        "cuda"
        if cuda_ok
        else "cpu"
    )

    print("PyTorch:", torch.__version__)
    print("CUDA available:", cuda_ok)

    if cuda_ok:
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )
        torch.cuda.reset_peak_memory_stats()

    # -------------------------------------------------------------------------
    # Clean 100k fitting/reference population
    # -------------------------------------------------------------------------

    X = np.load(
        require_file(
            root / "X.npy",
            "X.npy",
        ),
        mmap_mode="r",
        allow_pickle=False,
    )

    if X.shape != (
        EXPECTED_N,
        EXPECTED_D,
    ):
        raise ValueError(
            f"X.npy shape {X.shape}; "
            f"expected "
            f"{(EXPECTED_N, EXPECTED_D)}"
        )

    pool = np.load(
        require_file(
            baseline_dir
            / "Baseline_sample_indices.npy",
            "baseline fitting indices",
        ),
        allow_pickle=False,
    ).astype(
        np.int64,
        copy=False,
    ).reshape(-1)

    if (
        pool.size
        != BASELINE_FIT_SIZE
        or np.unique(
            pool
        ).size
        != pool.size
    ):
        raise ValueError(
            "Baseline_sample_indices.npy "
            "is not the expected unique "
            "300,000-row pool."
        )

    regenerated = (
        np.random.default_rng(
            EXPECTED_BASELINE_SUBSET_SEED
        ).choice(
            EXPECTED_N,
            size=BASELINE_FIT_SIZE,
            replace=False,
        ).astype(np.int64)
    )

    pool_matches_seed7 = bool(
        np.array_equal(
            pool,
            regenerated,
        )
    )

    if not pool_matches_seed7:
        raise ValueError(
            "Authoritative baseline pool "
            "does not match manuscript "
            "subset seed 7."
        )

    ref_idx = pool[
        :FINAL_TRAIN_N
    ].copy()

    np.save(
        out_dir
        / "INA_CR_reference_indices_100k.npy",
        ref_idx,
    )

    raw = np.asarray(
        X[ref_idx],
        dtype=np.float32,
    )

    print(
        f"Clean reference population: "
        f"{raw.shape}"
    )

    # -------------------------------------------------------------------------
    # Preprocessing + fixed INA-CR [0,1] mapping
    # -------------------------------------------------------------------------

    pre = ViewPreprocessor.load(
        clean_dir
    )

    project_views = pre.transform(
        raw
    )

    mm01 = MinMax01.fit(
        project_views
    )

    views = mm01.transform(
        project_views
    )

    save_input_ranges(
        out_dir,
        mm01,
    )

    print(
        "\nView audit after project "
        "preprocessing + INA-CR [0,1]:"
    )

    for name in VIEW_ORDER:
        arr = views[name]

        print(
            f"  {name:15s} "
            f"shape={arr.shape} "
            f"min={arr.min():.6f} "
            f"max={arr.max():.6f} "
            f"std={arr.std():.6f}"
        )

        if (
            arr.min() < -1e-7
            or arr.max() > 1.0000001
        ):
            raise ValueError(
                f"{name} outside [0,1]"
            )

    X01 = np.concatenate(
        [
            views[name]
            for name in VIEW_ORDER
        ],
        axis=1,
    ).astype(
        np.float32,
        copy=False,
    )

    if X01.shape != (
        FINAL_TRAIN_N,
        EXPECTED_D,
    ):
        raise ValueError(
            f"Unexpected X01 shape "
            f"{X01.shape}"
        )

    np.save(
        out_dir
        / "INA_CR_reference_X01_100k.npy",
        X01,
    )

    # -------------------------------------------------------------------------
    # Global kNN — clean reference only
    # -------------------------------------------------------------------------

    neighbor_path = (
        out_dir
        / "INA_CR_reference_neighbors_k6.npy"
    )

    if neighbor_path.is_file():
        print(
            "\nLoading saved clean-reference "
            "kNN matrix..."
        )

        global_neighbors = np.load(
            neighbor_path,
            allow_pickle=False,
        )

        if global_neighbors.shape != (
            FINAL_TRAIN_N,
            K_NEIBS,
        ):
            raise ValueError(
                "Saved neighbor matrix "
                "has unexpected shape."
            )

        knn_sec = None

    else:
        print(
            "\nBuilding clean-reference "
            "global kNN "
            "(ball_tree, k=6)..."
        )

        t_knn = time.time()

        nbrs = NearestNeighbors(
            n_neighbors=K_NEIBS + 1,
            algorithm="ball_tree",
        ).fit(X01)

        _, indices = nbrs.kneighbors(
            X01
        )

        global_neighbors = indices[
            :,
            1:K_NEIBS + 1,
        ].astype(
            np.int64,
            copy=False,
        )

        knn_sec = (
            time.time()
            - t_knn
        )

        np.save(
            neighbor_path,
            global_neighbors,
        )

        print(
            f"kNN complete in "
            f"{knn_sec:.3f} s; "
            f"matrix "
            f"{global_neighbors.shape}"
        )

    # -------------------------------------------------------------------------
    # Networks / datasets
    # -------------------------------------------------------------------------

    view_arrays = [
        views[name]
        for name in VIEW_ORDER
    ]

    view_tensors = [
        torch.from_numpy(arr)
        for arr in view_arrays
    ]

    datasets = [
        GeoViewDataset(
            arr,
            global_neighbors,
        )
        for arr in view_arrays
    ]

    encoders = [
        GeneratorEncode(
            VIEW_DIMS[name]
        )
        for name in VIEW_ORDER
    ]

    decoders = [
        GeneratorDecode(
            VIEW_DIMS[name]
        )
        for name in VIEW_ORDER
    ]

    predictors = [
        Prediction()
        for _ in VIEW_ORDER
    ]

    # -------------------------------------------------------------------------
    # Final training
    # -------------------------------------------------------------------------

    trainer = Trainer(
        device,
        out_dir,
    )

    t_train = time.time()

    (
        encoders,
        decoders,
        predictors,
    ) = trainer.train_all(
        datasets,
        view_tensors,
        encoders,
        decoders,
        predictors,
        resume=resume,
    )

    training_sec_this_process = (
        time.time()
        - t_train
    )

    # -------------------------------------------------------------------------
    # Save final trained model
    # -------------------------------------------------------------------------

    final_model_path = (
        out_dir
        / "INA_CR_final_model_100k_200epochs.pt"
    )

    atomic_torch_save(
        {
            "model_state": model_state(
                encoders,
                decoders,
                predictors,
            ),
            "view_order": VIEW_ORDER,
            "view_dims": VIEW_DIMS,
            "latent_dim": LATENT_DIM,
            "k": K_NEIBS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs_per_view": NUM_EPOCHS,
            "rounds": NUM_ROUNDS,
            "weights":
                list(WEIGHTS),
            "model_seed": MODEL_SEED,
        },
        final_model_path,
    )

    # -------------------------------------------------------------------------
    # Freeze clean-reference latent means
    # -------------------------------------------------------------------------

    print(
        "\nComputing frozen clean-reference "
        "latent means..."
    )

    latent_arrays, latent_means = (
        frozen_latent_means(
            encoders,
            views,
            device,
        )
    )

    np.savez(
        out_dir
        / "INA_CR_frozen_reference_latent_means.npz",
        **{
            name: latent_means[i]
            for i, name in enumerate(
                VIEW_ORDER
            )
        },
    )

    latent_diag = {
        name: {
            "mean": float(
                np.mean(
                    latent_arrays[i]
                )
            ),
            "std": float(
                np.std(
                    latent_arrays[i]
                )
            ),
        }
        for i, name in enumerate(
            VIEW_ORDER
        )
    }

    # -------------------------------------------------------------------------
    # Fit frozen clean-reference score-component ranges
    # -------------------------------------------------------------------------

    print(
        "Computing clean-reference "
        "inductive score ranges..."
    )

    t_score = time.time()

    ref_raw_scores = (
        score_raw_inductive(
            query_views=views,
            query_neighbor_idx=
                global_neighbors,
            ref_views=views,
            encoders=encoders,
            decoders=decoders,
            predictors=predictors,
            ref_latent_means=
                latent_means,
            device=device,
            chunk_size=1024,
        )
    )

    frozen_ranges = (
        fit_score_ranges(
            ref_raw_scores
        )
    )

    (
        aggregate_components,
        clean_total,
    ) = aggregate_frozen(
        ref_raw_scores,
        frozen_ranges,
    )

    score_range_sec = (
        time.time()
        - t_score
    )

    if not np.isfinite(
        clean_total
    ).all():
        raise FloatingPointError(
            "Non-finite final clean "
            "reference score."
        )

    save_component_ranges(
        out_dir,
        frozen_ranges,
    )

    np.save(
        out_dir
        / "INA_CR_clean_reference_total_scores.npy",
        clean_total.astype(
            np.float32
        ),
    )

    # -------------------------------------------------------------------------
    # Diagnostics / frozen protocol summary
    # -------------------------------------------------------------------------

    component_summary = {
        comp: {
            "min": float(
                np.min(arr)
            ),
            "max": float(
                np.max(arr)
            ),
            "mean": float(
                np.mean(arr)
            ),
            "std": float(
                np.std(arr)
            ),
        }
        for comp, arr in
        aggregate_components.items()
    }

    peak_alloc_mb = None
    peak_reserved_mb = None

    if cuda_ok:
        peak_alloc_mb = float(
            torch.cuda.max_memory_allocated()
            / (1024 ** 2)
        )

        peak_reserved_mb = float(
            torch.cuda.max_memory_reserved()
            / (1024 ** 2)
        )

    summary = {
        "stage":
            "INA_CR_final_clean_training",
        "status": "COMPLETE",
        "final_test_labels_used":
            False,
        "injected_data_used":
            False,
        "project_root":
            str(root),
        "output_dir":
            str(out_dir),
        "reference_count":
            FINAL_TRAIN_N,
        "reference_source":
            (
                "first 100,000 rows of "
                "authoritative 300,000-row "
                "baseline fitting pool"
            ),
        "baseline_pool_matches_seed_7":
            pool_matches_seed7,
        "view_order":
            VIEW_ORDER,
        "view_dims":
            VIEW_DIMS,
        "latent_dim":
            LATENT_DIM,
        "encoder_architecture":
            "Linear(D_v,32)+Tanh",
        "decoder_architecture":
            "Linear(32,D_v)+Sigmoid",
        "prediction_architecture":
            "Linear(32,32)+Tanh",
        "k_neighbors":
            K_NEIBS,
        "knn_space":
            (
                "concatenated 23-D "
                "[0,1] clean-reference "
                "representation"
            ),
        "knn_algorithm":
            "ball_tree",
        "batch_size":
            BATCH_SIZE,
        "learning_rate":
            LEARNING_RATE,
        "optimizer":
            "Adam AMSGrad",
        "weight_decay":
            WEIGHT_DECAY,
        "epochs_per_view":
            NUM_EPOCHS,
        "rounds":
            NUM_ROUNDS,
        "weights_rec_nei_pred_ins":
            list(WEIGHTS),
        "model_seed":
            MODEL_SEED,
        "project_preprocessing":
            True,
        "inacr_specific_scaling":
            (
                "feature-wise clean-reference "
                "min-max to [0,1], clipped"
            ),
        "decoder_output":
            "sigmoid",
        "inductive_scoring":
            True,
        "query_neighbors":
            (
                "k nearest frozen clean-"
                "reference rows"
            ),
        "test_batch_means_replaced_by":
            (
                "frozen clean-reference "
                "latent means"
            ),
        "released_batch_size_coefficient_retained":
            BATCH_SIZE,
        "component_normalization":
            (
                "frozen per-view/per-component "
                "clean-reference min/max; "
                "query score extrapolation "
                "not clipped"
            ),
        "knn_seconds_if_built_this_run":
            knn_sec,
        "training_seconds_this_process":
            training_sec_this_process,
        "clean_reference_scoring_seconds":
            score_range_sec,
        "clean_total_score":
            {
                "min":
                    float(
                        np.min(
                            clean_total
                        )
                    ),
                "max":
                    float(
                        np.max(
                            clean_total
                        )
                    ),
                "mean":
                    float(
                        np.mean(
                            clean_total
                        )
                    ),
                "std":
                    float(
                        np.std(
                            clean_total
                        )
                    ),
            },
        "aggregate_component_diagnostics":
            component_summary,
        "latent_diagnostics":
            latent_diag,
        "cuda_available":
            cuda_ok,
        "gpu":
            (
                torch.cuda.get_device_name(0)
                if cuda_ok
                else None
            ),
        "torch_version":
            torch.__version__,
        "peak_cuda_allocated_mb":
            peak_alloc_mb,
        "peak_cuda_reserved_mb":
            peak_reserved_mb,
        "saved_artifacts": {
            "final_model":
                final_model_path.name,
            "reference_indices":
                "INA_CR_reference_indices_100k.npy",
            "reference_X01":
                "INA_CR_reference_X01_100k.npy",
            "reference_neighbors":
                "INA_CR_reference_neighbors_k6.npy",
            "input_minmax":
                "INA_CR_input_minmax01_ranges.npz",
            "latent_means":
                "INA_CR_frozen_reference_latent_means.npz",
            "component_ranges_csv":
                "INA_CR_frozen_component_ranges.csv",
            "component_ranges_npz":
                "INA_CR_frozen_component_ranges.npz",
            "clean_reference_scores":
                "INA_CR_clean_reference_total_scores.npy",
            "loss_log":
                "INA_CR_training_loss_log.csv",
        },
        "note":
            (
                "Protocol frozen before final "
                "injected-test evaluation. "
                "No injected data, AUC, F1, "
                "or detection-rate values were "
                "used in training or parameter "
                "selection."
            ),
    }

    write_json(
        out_dir
        / "INA_CR_Final_Training_Summary.json",
        summary,
    )

    # The checkpoint is retained deliberately:
    # it documents the completed state and can
    # help recovery/audit. Do not use it as the
    # final model; use final_model_path.

    print("\n" + "=" * 80)
    print(
        "INA-CR FINAL TRAINING RESULT: COMPLETE"
    )
    print(
        f"Reference population: "
        f"{FINAL_TRAIN_N:,}"
    )
    print(
        f"Epochs per view: "
        f"{NUM_EPOCHS}"
    )
    print(
        f"Clean score: "
        f"min={clean_total.min():.6f}, "
        f"max={clean_total.max():.6f}, "
        f"mean={clean_total.mean():.6f}, "
        f"std={clean_total.std():.6f}"
    )
    print(
        f"Reference-score fitting time: "
        f"{score_range_sec:.2f} s"
    )
    print(
        "Final model:",
        final_model_path,
    )
    print(
        "Summary:",
        out_dir
        / "INA_CR_Final_Training_Summary.json",
    )
    print(
        "No injected data or final-test "
        "labels were used."
    )
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help=(
            "Project Code folder "
            "containing X.npy"
        ),
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Resume from the saved "
            "per-epoch checkpoint."
        ),
    )

    args = parser.parse_args()

    main(
        cli_root=args.root,
        resume=args.resume,
    )
