import torch
from torch import nn
import torch.nn.functional as functional


def initialize_saia(model, in_channels):
    """Initialize Spatially Adaptive Instance Aggregation (SAIA).

    SAIA operates directly on the active R101-C4 res4 tensor before the RPN.
    The standard Detectron2 R101-C4 res4 tensor has 1024 channels, so the
    experiment implementation uses an in_channels -> 256 -> in_channels
    bottleneck with no synthetic channel expansion.
    """
    import mmdet.models.utils
    from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence, build_positional_encoding
    from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
    model.lateral_conv_in = nn.Conv2d(in_channels, 256, 1)
    model.lateral_conv_out = nn.Conv2d(256, in_channels, 1)
    model.encoder = build_transformer_layer_sequence(dict(
        type="DetrTransformerEncoder", num_layers=4,
        transformerlayers=dict(type="BaseTransformerLayer", attn_cfgs=dict(
            type="MultiScaleDeformableAttention", embed_dims=256, num_levels=1),
            feedforward_channels=1024, ffn_dropout=0.1,
            operation_order=("self_attn", "norm", "ffn", "norm"))))
    model.positional_encoding = build_positional_encoding(dict(type="SinePositionalEncoding", num_feats=128, normalize=True))
    model.level_embeds = nn.Parameter(torch.empty(1, 256))
    model.reference_points = nn.Linear(256, 2)
    for module in (model.lateral_conv_in, model.lateral_conv_out, model.encoder, model.reference_points):
        for parameter in module.parameters():
            if parameter.ndim > 1:
                nn.init.xavier_uniform_(parameter)
        for child in module.modules():
            if isinstance(child, MultiScaleDeformableAttention):
                child.init_weights()
    nn.init.normal_(model.level_embeds)
    nn.init.zeros_(model.lateral_conv_in.bias)
    nn.init.zeros_(model.lateral_conv_out.bias)


def apply_saia(model, features, images):
    hidden = model.lateral_conv_in(features)
    batch, channels, height, width = hidden.shape
    # Match the feature-level validity convention used by the completed EADA
    # experiments: the complete active C4 grid participates in SAIA.  This is
    # intentionally not reconstructed from per-image padding extents, because
    # doing so changes the deformable-attention reference geometry relative to
    # the checkpoints used for the paper.
    mask = torch.zeros(
        (batch, height, width), dtype=torch.bool, device=hidden.device
    )
    valid_height = (~mask[:, :, 0]).sum(1).clamp_min(1)
    valid_width = (~mask[:, 0, :]).sum(1).clamp_min(1)
    ratios = torch.stack((valid_width / width, valid_height / height), -1)[:, None]
    grid_y, grid_x = torch.meshgrid(torch.arange(height, device=hidden.device) + 0.5, torch.arange(width, device=hidden.device) + 0.5, indexing="ij")
    reference = torch.stack((grid_x.flatten()[None] / valid_width[:, None], grid_y.flatten()[None] / valid_height[:, None]), -1)
    reference = reference[:, :, None] * ratios[:, None]
    position = model.positional_encoding(mask).flatten(2).permute(2, 0, 1) + model.level_embeds[0]
    encoded = model.encoder(
        query=hidden.flatten(2).permute(2, 0, 1), key=None, value=None,
        query_pos=position, query_key_padding_mask=mask.flatten(1),
        spatial_shapes=torch.tensor([[height, width]], device=hidden.device),
        reference_points=reference, level_start_index=torch.zeros(1, dtype=torch.long, device=hidden.device), valid_ratios=ratios)
    enhanced = model.lateral_conv_out(encoded.permute(1, 2, 0).reshape(batch, channels, height, width))
    return (1 - model.saia_alpha) * features + model.saia_alpha * enhanced
