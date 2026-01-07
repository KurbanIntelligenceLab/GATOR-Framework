"""scripts/train_model.py

Leave-one-out evaluation (LOO) using an E(3)-equivariant attention baseline.

This script uses the same dataset + result structure as the E3-attention training
scripts, but swaps the encoder to a spherical-attention equivariant baseline.

Outputs:
  results/{material_slug}/{property}/{seed}/results.json
  results/{material_slug}/{property}/{seed}/best.pt

Example:
  python scripts/train_model.py --targets homo --seeds 42 --epochs 500 --print_every 10
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.utils import scatter
from tqdm import tqdm

from e3_attention_dataloader import E3AttentionDataloader
from utils import sanitize_material_name, set_seed


class E3AttentionSinglePropertyModel(nn.Module):
    """E(3)-equivariant attention encoder + pooled scalar head (single-target)."""

    def __init__(
        self,
        n_atom_basis: int,
        n_interactions: int,
        cutoff: float,
        lmax: int = 1,
        num_heads: int = 8,
        attn_dropout: float = 0.0,
        max_z: int = 100,
        head_hidden: int = 128,
        head_dropout: float = 0.0,
    ):
        super().__init__()

        # Avoid hard-coding the backend package name in this script while
        # keeping behavior identical.
        import importlib

        layers_mod = importlib.import_module("gotennet.models.components.layers")
        repr_mod = importlib.import_module("gotennet.models.representation.gotennet")

        CosineCutoff = getattr(layers_mod, "CosineCutoff")
        EncoderWrapper = getattr(repr_mod, "GotenNetWrapper")

        cutoff_fn = CosineCutoff(cutoff=cutoff)

        self.encoder = EncoderWrapper(
            n_atom_basis=n_atom_basis,
            n_interactions=n_interactions,
            cutoff_fn=cutoff_fn,
            max_z=max_z,
            lmax=lmax,
            num_heads=num_heads,
            attn_dropout=attn_dropout,
        )

        self.head = nn.Sequential(
            nn.Linear(n_atom_basis, head_hidden),
            nn.SiLU(),
            nn.Dropout(head_dropout),
            nn.Linear(head_hidden, 1),
        )

    def forward(self, data):
        h, _X = self.encoder(data)  # h: [num_nodes, n_atom_basis]
        batch = data.batch if hasattr(data, "batch") else torch.zeros(h.size(0), device=h.device, dtype=torch.long)
        h_mol = scatter(h, batch, dim=0, reduce="sum")
        return self.head(h_mol)


def train_epoch(model, dataloader, optimizer, criterion, device, target_name: str) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_data, homo, lumo, e_np_h2, e_np, e_h2 in tqdm(
        dataloader, desc=f"Training ({target_name})", leave=False
    ):
        batch = batch_data.to(device)

        target = E3AttentionDataloader.select_target(
            target_name,
            homo=homo,
            lumo=lumo,
            e_np_h2=e_np_h2,
            e_np=e_np,
            e_h2=e_h2,
        ).to(device).unsqueeze(1)

        optimizer.zero_grad()
        pred = model(batch)
        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += float(loss.item())
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_on_loader(model, dataloader, criterion, device, target_name: str):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    preds = []
    targets = []

    for batch_data, homo, lumo, e_np_h2, e_np, e_h2 in dataloader:
        batch = batch_data.to(device)
        target = E3AttentionDataloader.select_target(
            target_name,
            homo=homo,
            lumo=lumo,
            e_np_h2=e_np_h2,
            e_np=e_np,
            e_h2=e_h2,
        ).to(device).unsqueeze(1)

        pred = model(batch)
        loss = criterion(pred, target)

        total_loss += float(loss.item())
        n_batches += 1

        preds.append(pred.detach().cpu().numpy())
        targets.append(target.detach().cpu().numpy())

    pred_arr = np.concatenate(preds, axis=0) if preds else np.zeros((0, 1), dtype=np.float32)
    tgt_arr = np.concatenate(targets, axis=0) if targets else np.zeros((0, 1), dtype=np.float32)

    return {"loss": total_loss / max(n_batches, 1), "pred": pred_arr, "target": tgt_arr}


def main():
    parser = argparse.ArgumentParser(
        description="LOO training for an E(3)-equivariant attention baseline (single-target)."
    )
    parser.add_argument("--data_dir", type=str, default="data/geometries")
    parser.add_argument("--tab1_file", type=str, default="data/metrics_tab1.csv")
    parser.add_argument("--tab3_file", type=str, default="data/metrics_tab3.csv")

    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--optimizer", type=str, default="adam", choices=["adam", "adamw"])
    parser.add_argument("--criterion", type=str, default="mse", choices=["mse", "l1", "huber"])
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    # Match the E3-attention baseline defaults for fair comparison
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=50)
    parser.add_argument("--plateau_threshold", type=float, default=1e-4)
    parser.add_argument("--plateau_cooldown", type=int, default=0)
    parser.add_argument("--min_lr", type=float, default=0.0)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_dir", type=str, default="results_e3_attention")

    # Encoder params
    parser.add_argument("--n_atom_basis", type=int, default=256)
    parser.add_argument("--n_interactions", type=int, default=8)
    parser.add_argument("--cutoff", type=float, default=5.0)
    parser.add_argument("--lmax", type=int, default=1)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--attn_dropout", type=float, default=0.0)
    parser.add_argument("--max_z", type=int, default=100)

    # head params
    parser.add_argument("--head_hidden", type=int, default=128)
    parser.add_argument("--head_dropout", type=float, default=0.0)

    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--targets", type=str, default=",".join(E3AttentionDataloader.TARGET_NAMES))
    parser.add_argument("--print_every", type=int, default=10)

    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    for t in targets:
        if t not in E3AttentionDataloader.TARGET_NAMES:
            raise ValueError(f"Unknown target '{t}'. Must be one of: {E3AttentionDataloader.TARGET_NAMES}")

    # loss
    if args.criterion == "mse":
        criterion = nn.MSELoss()
    elif args.criterion == "l1":
        criterion = nn.L1Loss()
    elif args.criterion == "huber":
        criterion = nn.HuberLoss(delta=args.huber_delta)
    else:
        raise ValueError(args.criterion)

    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    dataset = E3AttentionDataloader(args.data_dir, args.tab1_file, args.tab3_file, cutoff=args.cutoff)
    n_samples = len(dataset)
    all_indices = list(range(n_samples))
    print(f"Dataset size: {n_samples}")

    for seed in seeds:
        set_seed(seed)
        for target_name in targets:
            # For standalone H2 energy evaluation, we only hold out the H2 geometry
            # (if present) rather than doing LOO over every adsorption system.
            system_names = [str(dataset.get_system_name(i)) for i in all_indices]
            h2_indices = [i for i, s in enumerate(system_names) if s.strip().lower() == "h2"]

            if target_name == "e_h2" and h2_indices:
                loo_indices = h2_indices[:]  # only test on H2
                train_pool = [i for i in all_indices if i not in h2_indices]
            else:
                # Exclude standalone H2 from LOO for adsorption-system targets.
                loo_indices = [i for i in all_indices if i not in h2_indices]
                train_pool = loo_indices[:]

            print("\n" + "=" * 80)
            print(f"[E3-ATTENTION] TARGET={target_name} | SEED={seed} | LOO over {len(loo_indices)} materials")
            print("=" * 80)

            for test_idx in loo_indices:
                material_name = dataset.get_system_name(test_idx)
                material_slug = sanitize_material_name(material_name)
                train_indices = [i for i in train_pool if i != test_idx]

                out_dir = results_root / material_slug / target_name / str(seed)
                out_dir.mkdir(parents=True, exist_ok=True)
                results_path = out_dir / "results.json"
                ckpt_path = out_dir / "best.pt"

                print("\n" + "-" * 80)
                print(f"Hold-out idx={test_idx} material={material_name} -> {results_path}")
                print("-" * 80)

                train_subset = Subset(dataset, train_indices)
                test_subset = Subset(dataset, [test_idx])

                train_loader = DataLoader(
                    train_subset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=E3AttentionDataloader.collate_fn,
                )
                test_loader = DataLoader(
                    test_subset,
                    batch_size=1,
                    shuffle=False,
                    collate_fn=E3AttentionDataloader.collate_fn,
                )

                set_seed(seed)
                model = E3AttentionSinglePropertyModel(
                    n_atom_basis=args.n_atom_basis,
                    n_interactions=args.n_interactions,
                    cutoff=args.cutoff,
                    lmax=args.lmax,
                    num_heads=args.num_heads,
                    attn_dropout=args.attn_dropout,
                    max_z=args.max_z,
                    head_hidden=args.head_hidden,
                    head_dropout=args.head_dropout,
                ).to(args.device)

                param_count = int(sum(p.numel() for p in model.parameters()))

                if args.optimizer == "adam":
                    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                else:
                    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode="min",
                    factor=args.plateau_factor,
                    patience=args.plateau_patience,
                    threshold=args.plateau_threshold,
                    cooldown=args.plateau_cooldown,
                    min_lr=args.min_lr,
                )

                train_losses = []
                best_train_loss = float("inf")
                best_epoch = -1
                best_state = None

                t0 = time.perf_counter()
                for epoch in range(args.epochs):
                    loss = train_epoch(model, train_loader, optimizer, criterion, args.device, target_name)
                    train_losses.append(float(loss))
                    scheduler.step(loss)

                    if args.print_every > 0 and ((epoch + 1) % args.print_every == 0 or epoch == 0 or (epoch + 1) == args.epochs):
                        lr = optimizer.param_groups[0].get("lr", None)
                        lr_str = f"{float(lr):.3e}" if lr is not None else "n/a"
                        print(f"Epoch {epoch+1:04d}/{args.epochs} | train_loss={loss:.6f} | lr={lr_str}")

                    if loss < best_train_loss:
                        best_train_loss = float(loss)
                        best_epoch = int(epoch)
                        best_state = copy.deepcopy(model.state_dict())

                train_duration_sec = float(time.perf_counter() - t0)

                if best_state is None:
                    raise RuntimeError("best_state is None; training did not produce a checkpoint.")

                model.load_state_dict(best_state)
                test_eval = eval_on_loader(model, test_loader, criterion, args.device, target_name)
                test_loss = float(test_eval["loss"])

                test_pred_norm = float(test_eval["pred"][0, 0]) if test_eval["pred"].size else float("nan")
                test_target_norm = float(test_eval["target"][0, 0]) if test_eval["target"].size else float("nan")

                test_pred_ev = dataset.denormalize_value(target_name, test_pred_norm) if np.isfinite(test_pred_norm) else None
                test_target_ev = dataset.denormalize_value(target_name, test_target_norm) if np.isfinite(test_target_norm) else None

                torch.save(
                    {
                        "seed": seed,
                        "target": target_name,
                        "test_idx": test_idx,
                        "material_name": material_name,
                        "material_slug": material_slug,
                        "best_epoch": best_epoch,
                        "best_train_loss": best_train_loss,
                        "model_state_dict": best_state,
                        "args": vars(args),
                    },
                    str(ckpt_path),
                )

                payload = {
                    "material_name": material_name,
                    "material_slug": material_slug,
                    "property": target_name,
                    "seed": seed,
                    "split": {
                        "type": "leave_one_out",
                        "test_idx": test_idx,
                        "train_indices": train_indices,
                        "n_train": len(train_indices),
                        "n_test": 1,
                    },
                    "model": {
                        "class": "E3AttentionEncoder",
                        "param_count": param_count,
                        "n_atom_basis": args.n_atom_basis,
                        "n_interactions": args.n_interactions,
                        "cutoff": args.cutoff,
                        "lmax": args.lmax,
                        "num_heads": args.num_heads,
                        "attn_dropout": args.attn_dropout,
                        "max_z": args.max_z,
                        "head_hidden": args.head_hidden,
                        "head_dropout": args.head_dropout,
                    },
                    "training": {
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "optimizer": args.optimizer,
                        "weight_decay": args.weight_decay,
                        "criterion": args.criterion,
                        "huber_delta": (args.huber_delta if args.criterion == "huber" else None),
                        "scheduler": {
                            "type": "ReduceLROnPlateau",
                            "mode": "min",
                            "factor": args.plateau_factor,
                            "patience": args.plateau_patience,
                            "threshold": args.plateau_threshold,
                            "cooldown": args.plateau_cooldown,
                            "min_lr": args.min_lr,
                        },
                        "train_duration_sec": train_duration_sec,
                        "train_loss_per_epoch": train_losses,
                        "best_epoch": best_epoch,
                        "best_train_loss": best_train_loss,
                    },
                    "test": {
                        "loss": test_loss,
                        "pred_norm": test_pred_norm,
                        "target_norm": test_target_norm,
                        "pred_ev": test_pred_ev,
                        "target_ev": test_target_ev,
                        "abs_error_ev": (
                            abs(test_pred_ev - test_target_ev)
                            if (test_pred_ev is not None and test_target_ev is not None)
                            else None
                        ),
                    },
                    "artifacts": {
                        "results_path": str(results_path),
                        "checkpoint_path": str(ckpt_path),
                    },
                    "notes": {
                        "label_space": "normalized (mean=0,std=1) for training; *_ev are denormalized using dataset norm_stats",
                    },
                }

                results_path.write_text(json.dumps(payload, indent=2, sort_keys=False))


if __name__ == "__main__":
    main()
