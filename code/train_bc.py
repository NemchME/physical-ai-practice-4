"""Обучение BC-модели (CNN + MLP) на LeRobotDataset v3.0.

Запуск:
    python3 train_bc.py --train_dir dataset/train_1k --eval_dir dataset/eval --exp_name bc_1k --epochs 100 --batch_size 64 --lr 1e-4
    python3 train_bc.py --train_dir dataset/train_10k --eval_dir dataset/eval --exp_name bc_10k --epochs 100 --batch_size 64 --lr 1e-4
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from model import BCPolicy


def train_step(
    model,
    optimizer,
    obs_batch,
    action_batch,
    proprio_batch=None,
):
    """Один шаг обучения BC (см. Урок 5).

    TODO: реализуйте классический цикл PyTorch из 5 шагов:
      1) обнулите градиенты (``optimizer.zero_grad``);
      2) прямой проход: передайте в модель доступные ``obs_batch`` и
         ``proprio_batch`` именованными аргументами;
      3) посчитайте MSE-loss между ``pred`` и ``action_batch``;
      4) обратный проход (``loss.backward``);
      5) шаг оптимизатора (``optimizer.step``).
    Верните значение loss как число (``loss.item()``).
    """
    # TODO (Практика 4): proprio_batch может быть None в image-only режиме,
    # а obs_batch — в proprio-only; не подставляйте отсутствующую модальность.
    optimizer.zero_grad()
    if proprio_batch is None:
        pred = model(obs_batch)
    elif obs_batch is None:
        pred = model(proprio=proprio_batch)
    else:
        pred = model(obs=obs_batch, proprio=proprio_batch)
    loss = nn.functional.mse_loss(pred, action_batch)
    loss.backward()
    optimizer.step()
    return loss.item()


def parse_args():
    parser = argparse.ArgumentParser(description="Обучение BC-модели")
    parser.add_argument("--train_dir", type=str, required=True,
                        help="Корень тренировочного LeRobotDataset")
    parser.add_argument("--eval_dir", type=str, required=True,
                        help="Корень eval LeRobotDataset")
    parser.add_argument("--exp_name", type=str, required=True,
                        help="Название эксперимента (bc_1k, bc_10k)")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--obs_mode",
        choices=["image", "both", "proprio"],
        default="image",
        help="Наблюдения BC-политики: image, both или proprio",
    )
    parser.add_argument(
        "--proprio_encoder_type",
        choices=["mlp", "identity"],
        default="mlp",
        help="Ablation proprio: MLP или прямая конкатенация после нормализации",
    )
    return parser.parse_args()


class EpisodeDataset(Dataset):
    """Ленивая обёртка над LeRobotDataset.

    TODO (Практика 4):
      - примите obs_mode и проверьте одно из image/both/proprio;
      - читайте proprio только из sample["observation.proprio"];
      - возвращайте словарь с action и только нужными ключами image/proprio;
      - не ищите NPZ-файлы: источником остаётся LeRobotDataset v3.
    """

    def __init__(self, data_dir: str, obs_mode: str = "image"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        if obs_mode not in {"image", "both", "proprio"}:
            raise ValueError(f"Неизвестный obs_mode: {obs_mode}")
        self.obs_mode = obs_mode
        self.dataset = LeRobotDataset(
            repo_id="local/practice4",
            root=data_dir,
            video_backend="torchcodec",
        )
        self._proprio_cache = None
        self._action_cache = None
        if obs_mode == "proprio":
            hf_dataset = getattr(self.dataset, "hf_dataset", None)
            if hf_dataset is not None:
                plain = hf_dataset.with_format(None)
                self._proprio_cache = torch.tensor(
                    np.asarray(plain["observation.proprio"], dtype=np.float32)
                )
                self._action_cache = torch.tensor(
                    np.asarray(plain["action"], dtype=np.float32)
                )
        print(
            f"Loaded {self.dataset.meta.total_episodes} episodes, "
            f"{len(self)} steps from {data_dir}"
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if self._proprio_cache is not None:
            return {
                "proprio": self._proprio_cache[idx],
                "action": self._action_cache[idx],
            }
        sample = self.dataset[idx]
        item = {"action": sample["action"].to(torch.float32)}
        if self.obs_mode in ("image", "both"):
            item["image"] = sample["observation.images.front"].to(torch.float32)
        if self.obs_mode in ("both", "proprio"):
            item["proprio"] = sample["observation.proprio"].to(torch.float32)
        return item


def compute_proprio_stats(dataset):
    """Посчитать покомпонентные mean/std proprio только по train-датасету.

    TODO (Практика 4): объедините sample["proprio"] по всем кадрам,
    верните два float32-тензора формы (16,). Используйте population std
    (unbiased=False), чтобы результат не зависел от размера датасета.
    """
    values = None
    lerobot_ds = getattr(dataset, "dataset", None)
    hf_dataset = getattr(lerobot_ds, "hf_dataset", None)
    if hf_dataset is not None:
        try:
            column = hf_dataset.with_format(None)["observation.proprio"]
            values = torch.tensor(np.asarray(column, dtype=np.float32))
        except Exception:
            values = None
    if values is None:
        values = torch.stack([dataset[i]["proprio"].float() for i in range(len(dataset))])
    mean = values.mean(dim=0).to(torch.float32)
    std = values.std(dim=0, unbiased=False).to(torch.float32)
    return mean, std


def evaluate(model, eval_loader, criterion, device):
    """Посчитать eval loss для именованных мультимодальных batch.

    TODO (Практика 4): извлекайте image/proprio согласно флагам модели,
    переносите только доступные тензоры на device и вызывайте модель с тем же
    контрактом, что используется в train_step.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in eval_loader:
            obs = batch["image"].to(device) if model.use_image else None
            proprio = batch["proprio"].to(device) if model.use_proprio else None
            actions = batch["action"].to(device)
            pred = model(obs=obs, proprio=proprio)
            total_loss += criterion(pred, actions).item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    train_dataset = EpisodeDataset(args.train_dir, obs_mode=args.obs_mode)
    eval_dataset = EpisodeDataset(args.eval_dir, obs_mode=args.obs_mode)

    # TODO (Практика 4):
    #   - если режим использует proprio, вычислите mean/std только по train_dataset;
    #   - создайте BCPolicy с соответствующими use_image/use_proprio,
    #     статистиками и args.proprio_encoder_type.

    pin = (device == "cuda")
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=pin, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=pin,
    )

    use_image = args.obs_mode in ("image", "both")
    use_proprio = args.obs_mode in ("both", "proprio")
    proprio_mean, proprio_std = None, None
    if use_proprio:
        proprio_mean, proprio_std = compute_proprio_stats(train_dataset)
        print(f"Proprio mean: {proprio_mean.numpy().round(3)}")
        print(f"Proprio std:  {proprio_std.numpy().round(3)}")

    model = BCPolicy(
        action_dim=8,
        use_image=use_image,
        use_proprio=use_proprio,
        proprio_mean=proprio_mean,
        proprio_std=proprio_std,
        proprio_encoder_type=args.proprio_encoder_type,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    log_dir = os.path.join("logs", args.exp_name)
    ckpt_dir = os.path.join("logs", args.exp_name, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_eval_loss = float("inf")
    best_epoch = 0

    print(f"\nОбучение: {args.exp_name}")
    print(f"  Train: {len(train_dataset)} steps | Eval: {len(eval_dataset)} steps")
    print(f"  Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}")
    print(f"  obs_mode: {args.obs_mode} | proprio_encoder: {args.proprio_encoder_type}")
    print()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for batch in train_loader:
            # TODO (Практика 4): извлеките включённые модальности из batch,
            # перенесите их на device и передайте в train_step/model.
            obs = batch["image"].to(device) if use_image else None
            proprio = batch["proprio"].to(device) if use_proprio else None
            actions = batch["action"].to(device)
            loss_value = train_step(model, optimizer, obs, actions, proprio_batch=proprio)
            train_loss_sum += loss_value
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)
        eval_loss = evaluate(model, eval_loader, criterion, device)

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("eval/loss", eval_loss, epoch)

        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            best_epoch = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "eval_loss": eval_loss,
                "train_loss": train_loss,
                "obs_mode": args.obs_mode,
                "proprio_encoder_type": args.proprio_encoder_type,
            }, os.path.join(ckpt_dir, "best.pt"))

        elapsed = time.time() - t0
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss: {train_loss:.4f} | eval_loss: {eval_loss:.4f} | "
              f"{elapsed:.0f}s")

    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "eval_loss": eval_loss,
        "train_loss": train_loss,
        "obs_mode": args.obs_mode,
        "proprio_encoder_type": args.proprio_encoder_type,
    }, os.path.join(ckpt_dir, "last.pt"))

    writer.close()

    print()
    print(f"Checkpoint saved: {ckpt_dir}/last.pt")
    print(f"Best checkpoint: {ckpt_dir}/best.pt (epoch {best_epoch}, eval_loss={best_eval_loss:.4f})")


if __name__ == "__main__":
    main()
