"""BC-модель (CNN + MLP) и RL-политика (MLP) для PandaPickCube."""

import torch
import torch.nn as nn
import timm


class BCPolicy(nn.Module):
    """Мультимодальная BC-политика: image/proprio → вектор действия.

    TODO (Практика 4):
      - поддержите конфигурации image, both и proprio через флаги;
      - создавайте CNN только при use_image;
      - для proprio поддержите ablation ``mlp`` (16→64→64) и ``identity``
        (нормализованный вектор без обучаемого энкодера);
      - храните proprio mean/std через register_buffer;
      - стройте decoder от суммы размеров включённых веток.

    Для режима image сохраните имена модулей encoder/decoder и прежние размеры:
    это позволяет загружать legacy checkpoint без поля obs_mode.
    """

    def __init__(
        self,
        action_dim: int = 8,
        encoder_name: str = "resnet18",
        proprio_dim: int = 16,
        use_image: bool = True,
        use_proprio: bool = False,
        proprio_mean=None,
        proprio_std=None,
        proprio_encoder_type: str = "mlp",
    ):
        super().__init__()
        if not use_image and not use_proprio:
            raise ValueError("Нужно включить хотя бы одну модальность.")
        if proprio_encoder_type not in {"mlp", "identity"}:
            raise ValueError("proprio_encoder_type должен быть mlp или identity.")
        self.use_image = use_image
        self.use_proprio = use_proprio
        self.proprio_encoder_type = proprio_encoder_type
        self.proprio_dim = proprio_dim

        feature_dim = 0
        if use_image:
            self.encoder = timm.create_model(encoder_name, pretrained=True, num_classes=0)
            feature_dim += self.encoder.num_features

        if use_proprio:
            if proprio_mean is None:
                proprio_mean = torch.zeros(proprio_dim)
            if proprio_std is None:
                proprio_std = torch.ones(proprio_dim)
            self.register_buffer(
                "proprio_mean", torch.as_tensor(proprio_mean, dtype=torch.float32).clone()
            )
            self.register_buffer(
                "proprio_std", torch.as_tensor(proprio_std, dtype=torch.float32).clone()
            )
            if proprio_encoder_type == "mlp":
                self.proprio_encoder = nn.Sequential(
                    nn.Linear(proprio_dim, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                )
                feature_dim += 64
            else:
                self.proprio_encoder = nn.Identity()
                feature_dim += proprio_dim

        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs=None, proprio=None) -> torch.Tensor:
        """Выполнить forward по включённым модальностям.

        Изображение из LeRobot уже имеет формат BCHW float32 [0,1].

        TODO (Практика 4):
          1) для image-ветки проверьте BCHW-контракт и получите CNN-фичи;
          2) для proprio-ветки проверьте форму (B,16) и нормализуйте значения
             как (proprio - mean) / (std + 1e-6);
          3) в ablation ``mlp`` примените MLP 16→64→64, а в ``identity``
             передайте нормализованные 16 признаков напрямую;
          4) объедините доступные фичи через torch.cat(..., dim=-1);
          5) передайте результат в decoder.
        """
        features = []
        if self.use_image:
            if obs is None:
                raise ValueError("Модель ожидает изображение obs.")
            if obs.ndim == 3:
                obs = obs.unsqueeze(0)
            if obs.ndim != 4 or obs.shape[1] != 3:
                raise ValueError(
                    "Ожидалось RGB-изображение в формате (B,3,H,W), "
                    f"получена форма {tuple(obs.shape)}"
                )
            if obs.dtype != torch.float32:
                raise TypeError(
                    "Ожидался тензор float32 в диапазоне [0,1], "
                    f"получен тип {obs.dtype}"
                )
            features.append(self.encoder(obs))

        if self.use_proprio:
            if proprio is None:
                raise ValueError("Модель ожидает proprio.")
            if proprio.ndim == 1:
                proprio = proprio.unsqueeze(0)
            if proprio.ndim != 2 or proprio.shape[1] != self.proprio_dim:
                raise ValueError(
                    f"Ожидался proprio формы (B,{self.proprio_dim}), "
                    f"получена форма {tuple(proprio.shape)}"
                )
            proprio = (proprio.float() - self.proprio_mean) / (self.proprio_std + 1e-6)
            features.append(self.proprio_encoder(proprio))

        return self.decoder(torch.cat(features, dim=-1))


class RLPolicy(nn.Module):
    """MLP-политика для privileged state → action."""

    def __init__(self, state_dim: int = 29, action_dim: int = 8,
                 hidden_dims: tuple = (512, 256, 128)):
        super().__init__()
        layers = []
        in_dim = state_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), nn.ELU()])
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
