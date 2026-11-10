"""Rollout-оценка BC-моделей и RL-эксперта.

Запуск:
    python3 inference.py --checkpoint logs/bc_1k/checkpoints/best.pt --model bc --episodes 50 --seed 999
    python3 inference.py --checkpoint logs/bc_10k/checkpoints/best.pt --model bc --episodes 50 --seed 999
    python3 inference.py --checkpoint checkpoints/rl_expert.pt --model rl --episodes 50 --seed 999
"""

import argparse

import numpy as np
import torch

from env import PandaPickCubeEnv
from model import BCPolicy, RLPolicy


def parse_args():
    parser = argparse.ArgumentParser(description="Rollout-оценка BC/RL моделей")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Путь к чекпоинту модели")
    parser.add_argument("--model", type=str, required=True, choices=["bc", "rl"],
                        help="Тип модели: bc (визуомоторная) или rl (privileged state)")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Количество эпизодов (default: 50)")
    parser.add_argument("--seed", type=int, default=999,
                        help="Random seed (default: 999)")
    return parser.parse_args()


def load_bc_policy(checkpoint_path: str, device: str) -> BCPolicy:
    """Восстановить BCPolicy той же модальности, что использовалась в train.

    TODO (Практика 4):
      1) прочитайте obs_mode из checkpoint; для старых checkpoint используйте
         fallback "image";
      2) создайте BCPolicy с соответствующими use_image/use_proprio;
      3) восстановите proprio_encoder_type; для старых checkpoint используйте
         fallback "mlp";
      4) для proprio-режимов передайте временные mean=zeros(16), std=ones(16):
         реальные buffers восстановятся из model_state_dict;
      5) затем загрузите state_dict и переведите модель в eval.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    obs_mode = checkpoint.get("obs_mode", "image")
    proprio_encoder_type = checkpoint.get("proprio_encoder_type", "mlp")
    use_image = obs_mode in ("image", "both")
    use_proprio = obs_mode in ("both", "proprio")
    kwargs = {}
    if use_proprio:
        kwargs["proprio_mean"] = torch.zeros(16)
        kwargs["proprio_std"] = torch.ones(16)
    policy = BCPolicy(
        action_dim=8,
        use_image=use_image,
        use_proprio=use_proprio,
        proprio_encoder_type=proprio_encoder_type,
        **kwargs,
    )
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def load_rl_policy(checkpoint_path: str, device: str) -> RLPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    state_dim = checkpoint.get("state_dim", 29)
    action_dim = checkpoint.get("action_dim", 8)
    hidden_dims = checkpoint.get("hidden_dims", (512, 256, 128))

    policy = RLPolicy(state_dim=state_dim, action_dim=action_dim, hidden_dims=hidden_dims)
    policy.load_state_dict(checkpoint["model_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def run_episode_bc(env, policy, device, seed):
    """Один эпизод closed-loop инференса мультимодальной BC-модели.

    API среды этого проекта (не gym-стайл):
      ``obs = env.reset(seed=seed)``            — кадр HWC uint8 [0,255];
      ``obs, success, done = env.step(action)`` — три значения.

    TODO (Практика 4): реализуйте цикл на ``range(env.episode_length)``:
      1) если ``policy.use_image``, преобразуйте текущий HWC uint8 кадр
         в BCHW float32 [0,1];
      2) если ``policy.use_proprio``, вызовите ``env.get_proprio()`` для того же
         текущего состояния и добавьте batch-размерность;
      3) под ``torch.no_grad()`` вызовите ``policy(obs=..., proprio=...)``;
      4) выполните ``env.step`` и завершите цикл при ``done``.

    Не читайте LeRobot на инференсе: обе модальности приходят напрямую из env.
    Кадр и proprio на каждом шаге должны быть синхронны.
    Если эпизод не завершился за ``env.episode_length`` шагов — верните
    ``(False, env.episode_length)``.
    """
    obs = env.reset(seed=seed)
    for step in range(env.episode_length):
        obs_tensor = None
        proprio_tensor = None
        if policy.use_image:
            obs_tensor = (
                torch.from_numpy(obs)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
                .div(255.0)
            )
        if policy.use_proprio:
            proprio_tensor = (
                torch.from_numpy(env.get_proprio())
                .unsqueeze(0)
                .to(device=device, dtype=torch.float32)
            )
        with torch.no_grad():
            action = policy(obs=obs_tensor, proprio=proprio_tensor)
        action = action.squeeze(0).cpu().numpy()
        obs, success, done = env.step(action)
        if done:
            return success, step + 1
    return False, env.episode_length


def run_episode_rl(env, policy, device, seed):
    env.reset(seed=seed)
    state = env.get_privileged_state()

    for step in range(env.episode_length):
        state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = policy(state_tensor).squeeze(0).cpu().numpy()
        _, success, done = env.step(action)
        state = env.get_privileged_state()
        if done:
            return success, step + 1

    return False, env.episode_length


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Model: {args.model}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Episodes: {args.episodes}")
    print(f"Seed: {args.seed}")
    print()

    if args.model == "bc":
        policy = load_bc_policy(args.checkpoint, device)
        run_fn = run_episode_bc
    else:
        policy = load_rl_policy(args.checkpoint, device)
        run_fn = run_episode_rl

    env = PandaPickCubeEnv()
    rng = np.random.RandomState(args.seed)

    successes = 0
    for ep in range(args.episodes):
        ep_seed = rng.randint(0, 2**31)
        success, steps = run_fn(env, policy, device, ep_seed)
        diagnostic = env.get_rollout_diagnostic()
        status = "success" if success else f"fail — {diagnostic}"
        successes += int(success)
        print(f"Episode {ep + 1}/{args.episodes}: {status} ({steps} steps)")

    sr = successes / args.episodes * 100
    print(f"\nSuccess rate: {successes}/{args.episodes} ({sr:.1f}%)")

    env.close()


if __name__ == "__main__":
    main()
