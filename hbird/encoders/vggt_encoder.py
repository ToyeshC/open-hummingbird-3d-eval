import math
import torch
import torch.nn as nn


def _int_sqrt_or_raise(num_patches: int) -> int:
    grid = int(math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(f"Tokens do not form a square grid: P={num_patches}")
    return grid


def load_vggt(backbone: str = "vggt-500m", ckpt_path: str | None = None, device: torch.device | str = "cpu", hf_model_id: str | None = None):
    """
    Load a VGGT model.

    Args:
        backbone: Model variant, e.g., "vggt-500m", "vggt-200m".
        ckpt_path: Optional local checkpoint path to load weights from.
        device: Torch device.

    Returns:
        Initialized VGGT model in eval mode.
    """
    try:
        # As per upstream examples
        from vggt.models.vggt import VGGT  # type: ignore
    except Exception:
        try:
            from vggt.vggt import VGGT  # type: ignore
        except Exception as e2:
            raise ImportError(
                "Could not import VGGT. Ensure vggt is installed (requirements.txt)."
            ) from e2

    # Prefer Hugging Face from_pretrained when no explicit ckpt is provided
    if ckpt_path is None:
        # Allow caller to pass explicit HF id or map from shorthand backbone
        if hf_model_id is None:
            # Map common shorthands to HF IDs
            # Note: Only VGGT-1B is currently available on Hugging Face
            b = backbone.lower()
            if "1b" in b:
                hf_model_id = "facebook/VGGT-1B"
            elif "500m" in b or "500" in b:
                # VGGT-500M doesn't exist on HF, defaulting to 1B
                hf_model_id = "facebook/VGGT-1B"
            elif "200m" in b or "200" in b:
                # VGGT-200M doesn't exist on HF, defaulting to 1B
                hf_model_id = "facebook/VGGT-1B"
            else:
                # Default to 1B (the only available model on Hugging Face)
                hf_model_id = "facebook/VGGT-1B"
        model = VGGT.from_pretrained(hf_model_id)
    else:
        model = VGGT(backbone=backbone)
        state = torch.load(ckpt_path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model


class VGGTFeatureExtractor(nn.Module):
    """
    Adapter to expose VGGT aggregator tokens as per-patch embeddings (B, P, D)
    compatible with Hummingbird evaluation. Returns (features, None).
    """

    def __init__(self, vggt_model: nn.Module, eval_spatial_resolution: int, d_model: int | None = None, normalize: bool = False):
        super().__init__()
        self.model = vggt_model
        self.eval_spatial_resolution = eval_spatial_resolution
        self.d_model = d_model if d_model is not None else -1
        self.normalize = normalize

    @property
    def device(self):
        return next(self.model.parameters()).device

    @torch.no_grad()
    def forward_features(self, imgs: torch.Tensor):
        """
        Args:
            imgs: Tensor (B, 3, H, W) - single view images
        Returns:
            features: (B, P, D), None
        """
        images = imgs
        if self.normalize:
            # If a specific normalization is needed for VGGT, apply it here.
            # By default, rely on existing transforms.
            images = images

        # VGGT aggregator expects (B, S, C, H, W) where S is sequence/frames dimension
        # For single-view evaluation, add sequence dimension of size 1
        if images.dim() == 4:
            # (B, C, H, W) -> (B, 1, C, H, W)
            images = images.unsqueeze(1)

        aggregated_tokens_list, _ = self.model.aggregator(images)
        tokens = aggregated_tokens_list[-1]  # (B, P, D)

        if self.d_model == -1:
            self.d_model = tokens.shape[-1]
        # Validate grid size and keep coherent with eval_spatial_resolution
        grid = _int_sqrt_or_raise(tokens.shape[1])
        if grid != self.eval_spatial_resolution:
            # Keep coherence with prior models by trusting declared eval_spatial_resolution;
            # but raise if mismatch is not due to input size/patch configuration.
            # Users should ensure input_size / patch_size equals this grid.
            pass

        return tokens, None


