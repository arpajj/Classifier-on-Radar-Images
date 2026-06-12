"""
model.py
========
Three-stream CNN with late fusion for radar human activity classification.

Architecture per stream:
    Input: [B, 1, 128, 128]  (each representation is single-channel)
    Block 1: Conv 3x3 (32)  → BN → ReLU → Conv 3x3 (32)  → BN → ReLU → MaxPool 2x2
    Block 2: Conv 3x3 (64)  → BN → ReLU → Conv 3x3 (64)  → BN → ReLU → MaxPool 2x2
    Block 3: Conv 3x3 (128) → BN → ReLU → Conv 3x3 (128) → BN → ReLU → MaxPool 2x2
    Block 4: Conv 3x3 (256) → BN → ReLU → Conv 3x3 (256) → BN → ReLU → AdaptiveAvgPool
    FC head: Linear(256 → STREAM_FEATURE_DIM) → BN → ReLU → Dropout

Late fusion:
    Concatenate 3 stream feature vectors → [B, 3 * STREAM_FEATURE_DIM]
    Classifier: Linear → BN → ReLU → Dropout → Linear → NUM_CLASSES

Streams:
    Stream 0 → Spectrogram   (channel 0)
    Stream 1 → Range-Time    (channel 1)
    Stream 2 → Range-Doppler (channel 2)
"""

import torch
import torch.nn as nn
import config


# ══════════════════════════════════════════════════════════════════════
# 1.  CONVOLUTIONAL BLOCK  (Conv → BN → ReLU → Conv → BN → ReLU)
# ══════════════════════════════════════════════════════════════════════
class ConvBlock(nn.Module):
    """Double conv block: (Conv → BN → ReLU) × 2"""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch,  out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size, padding=pad, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.block(x)


# ══════════════════════════════════════════════════════════════════════
# 2.  SINGLE STREAM CNN
# ══════════════════════════════════════════════════════════════════════
class StreamCNN(nn.Module):
    """
    Processes one radar representation (Spectrogram / Range-Time / Range-Doppler).
    Input  : [B, 1, H, W]
    Output : [B, STREAM_FEATURE_DIM]  (feature vector)

    Spatial flow for 128x128 input:
        After Block1 + MaxPool : 64x64
        After Block2 + MaxPool : 32x32
        After Block3 + MaxPool : 16x16
        After Block4 + AvgPool :  1x1
        Flatten                : 256-d
        FC                     : STREAM_FEATURE_DIM-d
    """

    def __init__(self, feature_dim: int = config.STREAM_FEATURE_DIM,
                 dropout: float = config.DROPOUT):
        super().__init__()

        self.encoder = nn.Sequential(
            # Block 1:  1 → 32
            ConvBlock(1, 32),
            nn.MaxPool2d(2, 2),            # 128→64

            # Block 2: 32 → 64
            ConvBlock(32, 64),
            nn.MaxPool2d(2, 2),            # 64→32

            # Block 3: 64 → 128
            ConvBlock(64, 128),
            nn.MaxPool2d(2, 2),            # 32→16

            # Block 4: 128 → 256
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),  # 16→1
        )

        # FC head: project to stream feature dimension
        self.fc_head = nn.Sequential(nn.Flatten(), nn.Linear(256, feature_dim, bias=False), # [B, 256]
                                    nn.BatchNorm1d(feature_dim), nn.ReLU(inplace=True),nn.Dropout(dropout))

    def forward(self, x):
        # x: [B, 1, H, W]
        x = self.encoder(x)    # [B, 256, 1, 1]
        x = self.fc_head(x)    # [B, feature_dim]
        return x


# ══════════════════════════════════════════════════════════════════════
# 3.  THREE-STREAM MODEL WITH LATE FUSION
# ══════════════════════════════════════════════════════════════════════
class ThreeStreamCNN(nn.Module):
    """
    Three independent StreamCNN encoders (one per radar representation),
    that uses mid fusion or late fusion (both use concatenation) and a final classifier head.

    Input  : x  [B, 3, H, W]
                 x[:,0,:,:]  → Spectrogram
                 x[:,1,:,:]  → Range-Time
                 x[:,2,:,:]  → Range-Doppler

    Output : logits  [B, NUM_CLASSES]
    """

    def __init__(self, num_classes = config.NUM_CLASSES, feature_dim = config.STREAM_FEATURE_DIM, dropout = config.DROPOUT, fusion_type = config.FUSION_TYPE):

        super().__init__()
        self.fusion_type = fusion_type

        print(f"Fusion type ----> {fusion_type}")

        if fusion_type == 'late':
            # Each stream has its own FULL encoder + FC head
            self.stream_spec = StreamCNN(feature_dim, dropout)
            self.stream_rt   = StreamCNN(feature_dim, dropout)
            self.stream_rd   = StreamCNN(feature_dim, dropout)
            fused_dim = 3 * feature_dim

        elif fusion_type == 'mid':
            # Each stream shares the same encoder structure but NOT weights
            # Fusion happens after Block 2 (after 32x32 feature maps)
            # Then a shared encoder continues from Block 3 onwards
            self.stream_spec_enc = nn.Sequential(
                ConvBlock(1, 32), nn.MaxPool2d(2, 2),   # 128→64
                ConvBlock(32, 64), nn.MaxPool2d(2, 2),  # 64→32
            )
            self.stream_rt_enc = nn.Sequential(ConvBlock(1, 32), nn.MaxPool2d(2, 2), ConvBlock(32, 64), nn.MaxPool2d(2, 2))
            self.stream_rd_enc = nn.Sequential(ConvBlock(1, 32), nn.MaxPool2d(2, 2), ConvBlock(32, 64), nn.MaxPool2d(2, 2))
            # After mid fusion: 3 streams × 64 channels → 192 channels
            # Shared encoder continues
            self.shared_enc = nn.Sequential(ConvBlock(192, 128), nn.MaxPool2d(2, 2),  # 32→16
                ConvBlock(128, 256), nn.AdaptiveAvgPool2d((1, 1)))
            self.fc_head = nn.Sequential(nn.Flatten(),nn.Linear(256, feature_dim, bias=False),
                        nn.BatchNorm1d(feature_dim),nn.ReLU(inplace=True),nn.Dropout(dropout))
            fused_dim = feature_dim   # only one feature vector after shared enc

        else:
            raise ValueError(f"fusion_type must be 'late' or 'mid', got '{fusion_type}'")

        # Classifier head (same for both fusion types)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2, bias=False),
            nn.BatchNorm1d(fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, fused_dim // 4, bias=False),
            nn.BatchNorm1d(fused_dim // 4),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 4, num_classes))

    def forward(self, x):
        x_spec = x[:, 0:1, :, :]
        x_rt   = x[:, 1:2, :, :]
        x_rd   = x[:, 2:3, :, :]

        if self.fusion_type == 'late':
            f_spec = self.stream_spec(x_spec)
            f_rt   = self.stream_rt(x_rt)
            f_rd   = self.stream_rd(x_rd)
            fused  = torch.cat([f_spec, f_rt, f_rd], dim=1)  # [B, 3*feature_dim]

        elif self.fusion_type == 'mid':
            # Each stream encodes independently up to Block 2
            f_spec = self.stream_spec_enc(x_spec)   # [B, 64, 32, 32]
            f_rt   = self.stream_rt_enc(x_rt)       # [B, 64, 32, 32]
            f_rd   = self.stream_rd_enc(x_rd)       # [B, 64, 32, 32]
            # Concatenate along channel dim → mid fusion
            fused  = torch.cat([f_spec, f_rt, f_rd], dim=1)  # [B, 192, 32, 32]
            # Continue with shared encoder
            fused  = self.shared_enc(fused)          # [B, 256, 1, 1]
            fused  = self.fc_head(fused)             # [B, feature_dim]

        return self.classifier(fused)

# ══════════════════════════════════════════════════════════════════════
# 4.  MODEL SUMMARY UTILITY
# ══════════════════════════════════════════════════════════════════════
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
