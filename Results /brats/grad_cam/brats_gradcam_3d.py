"""
============================================================
BraTS 2021 - Memory-Efficient 3D Grad-CAM
============================================================

Purpose
-------
Visualize which regions of a 3D MRI patch contribute most
strongly to the whole-tumor segmentation prediction.

Strategy
--------
Because full-volume Grad-CAM is too memory-intensive for a
4 GB GPU, the analysis is performed in two stages:

1. Full-volume prediction using MONAI sliding-window inference.
2. A tumor-centered 96x96x96 patch is extracted.
3. Grad-CAM is computed only on that patch.

This preserves the full-volume prediction step while keeping
the gradient-based explainability stage GPU-efficient.

Model
-----
MONAI SegResNet

    spatial_dims = 3
    init_filters = 32
    in_channels = 4
    out_channels = 1
    dropout_prob = 0.2
    blocks_down = (1, 2, 2, 4)
    blocks_up = (1, 1, 1)

Input channels
--------------
    0 = T1
    1 = T1ce
    2 = T2
    3 = FLAIR

Checkpoint
----------
Epoch 25
Best validation Dice = 0.895904

Output
------
results/
└── gradcam/
    └── <patient_id>/
        ├── axial.png
        ├── coronal.png
        ├── sagittal.png
        ├── axial_diagnostic.png
        ├── coronal_diagnostic.png
        ├── sagittal_diagnostic.png
        ├── gradcam.npy
        ├── patch_prediction.npy
        ├── full_volume_probability.npy
        ├── ground_truth.npy
        └── metadata.txt
"""

# ============================================================
# IMPORTS
# ============================================================

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.inferers import sliding_window_inference
from monai.networks.nets import SegResNet
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


# ============================================================
# CONFIGURATION
# ============================================================

DATA_ROOT = Path(
    r"G:\CancerAI\Data\BraTS2021_Task1"
    r"\brats-2021-task1"
    r"\BraTS2021_Training_Data"
)

CHECKPOINT_PATH = Path(
    r"G:\CancerAI\Code\code\version4"
    r"\brats_pretraining_final"
    r"\checkpoints"
    r"\brats_segresnet_best.pth"
)

OUTPUT_ROOT = Path(
    r"G:\CancerAI\Code\code\version4"
    r"\brats_pretraining_final"
    r"\results"
    r"\gradcam"
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

# ------------------------------------------------------------
# Grad-CAM patch size
# ------------------------------------------------------------

PATCH_SIZE = (
    96,
    96,
    96,
)

# ------------------------------------------------------------
# Sliding-window inference
# ------------------------------------------------------------

SW_BATCH_SIZE = 1
SW_OVERLAP = 0.25

# ------------------------------------------------------------
# Tumor threshold
# ------------------------------------------------------------

THRESHOLD = 0.5

# ------------------------------------------------------------
# Number of patients
# ------------------------------------------------------------

NUM_PATIENTS = 1


# ============================================================
# LOGGING
# ============================================================

OUTPUT_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(
    __name__
)


# ============================================================
# PREPROCESSING
# ============================================================

def create_base_transform():
    """
    Exact deterministic preprocessing used during training.
    """

    return Compose(
        [

            LoadImaged(
                keys=[
                    "flair",
                    "t1",
                    "t1ce",
                    "t2",
                    "seg",
                ]
            ),

            EnsureChannelFirstd(
                keys=[
                    "flair",
                    "t1",
                    "t1ce",
                    "t2",
                    "seg",
                ]
            ),

            Orientationd(
                keys=[
                    "flair",
                    "t1",
                    "t1ce",
                    "t2",
                    "seg",
                ],
                axcodes="RAS",
            ),

            Spacingd(
                keys=[
                    "flair",
                    "t1",
                    "t1ce",
                    "t2",
                ],
                pixdim=(
                    1.0,
                    1.0,
                    1.0,
                ),
                mode="bilinear",
            ),

            Spacingd(
                keys=[
                    "seg",
                ],
                pixdim=(
                    1.0,
                    1.0,
                    1.0,
                ),
                mode="nearest",
            ),

            ScaleIntensityRangePercentilesd(
                keys=[
                    "flair",
                    "t1",
                    "t1ce",
                    "t2",
                ],
                lower=1,
                upper=99,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
        ]
    )


# ============================================================
# MODEL
# ============================================================

def create_model():

    model = SegResNet(
        spatial_dims=3,
        init_filters=32,
        in_channels=4,
        out_channels=1,
        dropout_prob=0.2,
        blocks_down=(
            1,
            2,
            2,
            4,
        ),
        blocks_up=(
            1,
            1,
            1,
        ),
    )

    return model


# ============================================================
# CHECKPOINT
# ============================================================

def load_model():

    logger.info("=" * 70)
    logger.info("LOADING BEST CHECKPOINT")
    logger.info("=" * 70)

    if not CHECKPOINT_PATH.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{CHECKPOINT_PATH}"
        )

    model = create_model()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        state_dict = (
            checkpoint[
                "model_state_dict"
            ]
        )

        logger.info(
            f"Checkpoint epoch: "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

        logger.info(
            f"Best validation Dice: "
            f"{checkpoint.get('best_val_dice', 'unknown')}"
        )

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith(
            "module."
        ):

            key = key[
                len("module.") :
            ]

        cleaned_state_dict[
            key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True,
    )

    model.to(
        DEVICE
    )

    model.eval()

    logger.info(
        "Checkpoint loaded successfully."
    )

    return model


# ============================================================
# PATIENT SCANNER
# ============================================================

def find_patients():

    if not DATA_ROOT.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n"
            f"{DATA_ROOT}"
        )

    patients = []

    patient_dirs = sorted(
        p
        for p in DATA_ROOT.iterdir()
        if p.is_dir()
    )

    for patient_dir in patient_dirs:

        patient_id = (
            patient_dir.name
        )

        paths = {
            "flair": (
                patient_dir /
                f"{patient_id}_flair.nii.gz"
            ),

            "t1": (
                patient_dir /
                f"{patient_id}_t1.nii.gz"
            ),

            "t1ce": (
                patient_dir /
                f"{patient_id}_t1ce.nii.gz"
            ),

            "t2": (
                patient_dir /
                f"{patient_id}_t2.nii.gz"
            ),

            "seg": (
                patient_dir /
                f"{patient_id}_seg.nii.gz"
            ),
        }

        if all(
            p.exists()
            for p in paths.values()
        ):

            patients.append(
                {
                    "patient_id": patient_id,
                    **{
                        key: str(value)
                        for key, value
                        in paths.items()
                    },
                }
            )

    logger.info(
        f"Complete patients found: "
        f"{len(patients)}"
    )

    return patients


# ============================================================
# PREPARE PATIENT
# ============================================================

def prepare_patient(
    patient,
    transform,
):

    data = transform(
        patient
    )

    image = torch.cat(
        [
            data["t1"],
            data["t1ce"],
            data["t2"],
            data["flair"],
        ],
        dim=0,
    ).float()

    ground_truth = (
        data["seg"] > 0
    ).float()

    image = image.unsqueeze(
        0
    )

    ground_truth = (
        ground_truth.unsqueeze(0)
    )

    return (
        image,
        ground_truth,
    )


# ============================================================
# PAD VOLUME
# ============================================================

def pad_volume(
    tensor: torch.Tensor,
    multiple: int = 16,
):
    """
    Pad [B,C,D,H,W] so spatial dimensions are divisible by 16.
    """

    if tensor.ndim != 5:

        raise ValueError(
            f"Expected 5D tensor, "
            f"received {tensor.shape}"
        )

    _, _, d, h, w = (
        tensor.shape
    )

    target_d = (
        (d + multiple - 1)
        // multiple
    ) * multiple

    target_h = (
        (h + multiple - 1)
        // multiple
    ) * multiple

    target_w = (
        (w + multiple - 1)
        // multiple
    ) * multiple

    pd = target_d - d
    ph = target_h - h
    pw = target_w - w

    padding = (
        0,
        pw,
        0,
        ph,
        0,
        pd,
    )

    if any(
        p > 0
        for p in (
            pd,
            ph,
            pw,
        )
    ):

        tensor = F.pad(
            tensor,
            padding,
            mode="constant",
            value=0.0,
        )

    return (
        tensor,
        (d, h, w),
    )


# ============================================================
# FULL-VOLUME SLIDING WINDOW INFERENCE
# ============================================================

@torch.no_grad()
def run_full_volume_inference(
    model,
    image,
):
    """
    Perform memory-efficient full-volume inference.

    No gradients are stored here.
    """

    padded_image, original_shape = (
        pad_volume(
            image,
            multiple=16,
        )
    )

    logger.info(
        f"Full-volume input: "
        f"{tuple(image.shape)}"
    )

    logger.info(
        f"Padded volume: "
        f"{tuple(padded_image.shape)}"
    )

    logger.info(
        "Running sliding-window inference..."
    )

    logits = sliding_window_inference(
        inputs=padded_image,
        roi_size=PATCH_SIZE,
        sw_batch_size=SW_BATCH_SIZE,
        predictor=model,
        overlap=SW_OVERLAP,
        mode="gaussian",
    )

    logits = logits[
        ...,
        :original_shape[0],
        :original_shape[1],
        :original_shape[2],
    ]

    probability = torch.sigmoid(
        logits
    )

    logger.info(
        f"Full-volume output: "
        f"{tuple(probability.shape)}"
    )

    return probability


# ============================================================
# FIND TUMOR CENTER
# ============================================================

def find_tumor_center(
    probability: torch.Tensor,
):
    """
    Find the centroid of the predicted tumor mask.

    Falls back to the global maximum probability if the
    thresholded mask is empty.
    """

    probability_3d = (
        probability[
            0,
            0
        ]
    )

    tumor_mask = (
        probability_3d >
        THRESHOLD
    )

    count = int(
        tumor_mask.sum().item()
    )

    logger.info(
        f"Predicted tumor voxels: "
        f"{count:,}"
    )

    if count == 0:

        logger.warning(
            "No tumor voxels above threshold."
        )

        index = torch.argmax(
            probability_3d
        )

        center = np.array(
            np.unravel_index(
                int(index.item()),
                probability_3d.shape,
            )
        )

    else:

        coords = (
            torch.nonzero(
                tumor_mask,
                as_tuple=False,
            )
            .detach()
            .cpu()
            .numpy()
        )

        center = coords.mean(
            axis=0
        )

        center = np.rint(
            center
        ).astype(
            int
        )

    logger.info(
        f"Tumor center: "
        f"{tuple(center.tolist())}"
    )

    return center


# ============================================================
# EXTRACT TUMOR-CENTERED PATCH
# ============================================================

def extract_centered_patch(
    tensor: torch.Tensor,
    center,
    patch_size: Tuple[int, int, int],
):
    """
    Extract a fixed-size patch around the supplied center.

    Tensor shape:
        [B,C,D,H,W]

    Returns:
        patch
        crop coordinates
    """

    _, _, d, h, w = (
        tensor.shape
    )

    pd, ph, pw = (
        patch_size
    )

    cd, ch, cw = (
        [int(v) for v in center]
    )

    start_d = (
        cd - pd // 2
    )

    start_h = (
        ch - ph // 2
    )

    start_w = (
        cw - pw // 2
    )

    start_d = max(
        0,
        min(
            start_d,
            d - pd,
        )
    )

    start_h = max(
        0,
        min(
            start_h,
            h - ph,
        )
    )

    start_w = max(
        0,
        min(
            start_w,
            w - pw,
        )
    )

    end_d = (
        start_d + pd
    )

    end_h = (
        start_h + ph
    )

    end_w = (
        start_w + pw
    )

    patch = tensor[
        :,
        :,
        start_d:end_d,
        start_h:end_h,
        start_w:end_w,
    ]

    logger.info(
        f"Patch coordinates: "
        f"D[{start_d}:{end_d}] "
        f"H[{start_h}:{end_h}] "
        f"W[{start_w}:{end_w}]"
    )

    logger.info(
        f"Patch shape: "
        f"{tuple(patch.shape)}"
    )

    return (
        patch,
        (
            start_d,
            start_h,
            start_w,
        ),
    )


# ============================================================
# INSERT PATCH INTO FULL VOLUME
# ============================================================

def place_patch_in_volume(
    patch_map: np.ndarray,
    volume_shape,
    coordinates,
):
    """
    Place a patch-sized map back into the full-volume
    coordinate system.
    """

    start_d, start_h, start_w = (
        coordinates
    )

    pd, ph, pw = (
        patch_map.shape
    )

    output = np.zeros(
        volume_shape,
        dtype=np.float32,
    )

    output[
        start_d:
        start_d + pd,

        start_h:
        start_h + ph,

        start_w:
        start_w + pw,
    ] = patch_map

    return output


# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM3D:

    def __init__(
        self,
        model,
        target_layer,
    ):

        self.model = model
        self.target_layer = (
            target_layer
        )

        self.activations = None
        self.gradients = None

        self.forward_handle = (
            target_layer.register_forward_hook(
                self.forward_hook
            )
        )

        self.backward_handle = (
            target_layer.register_full_backward_hook(
                self.backward_hook
            )
        )

    def forward_hook(
        self,
        module,
        inputs,
        output,
    ):

        self.activations = output

    def backward_hook(
        self,
        module,
        grad_input,
        grad_output,
    ):

        self.gradients = (
            grad_output[0]
        )

    def remove_hooks(self):

        self.forward_handle.remove()
        self.backward_handle.remove()

    def generate(
        self,
        image,
    ):

        self.model.zero_grad(
            set_to_none=True
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # image is only 96x96x96 here.
        # ----------------------------------------------------

        logits = self.model(
            image
        )

        if not torch.is_tensor(
            logits
        ):

            raise RuntimeError(
                "Model output is not a tensor."
            )

        if logits.ndim != 5:

            raise RuntimeError(
                f"Expected 5D output, "
                f"got {logits.shape}"
            )

        if logits.shape[1] != 1:

            raise RuntimeError(
                "Expected binary segmentation "
                "with one output channel."
            )

        probability = torch.sigmoid(
            logits
        )

        probability_3d = (
            probability[
                0,
                0
            ]
        )

        tumor_mask = (
            probability_3d >
            THRESHOLD
        )

        tumor_voxels = int(
            tumor_mask.sum().item()
        )

        logger.info(
            f"Patch tumor voxels: "
            f"{tumor_voxels:,}"
        )

        if tumor_voxels > 0:

            target = (
                probability_3d[
                    tumor_mask
                ].mean()
            )

        else:

            logger.warning(
                "No tumor prediction in Grad-CAM patch."
            )

            target = (
                probability_3d.max()
            )

        logger.info(
            f"Grad-CAM target score: "
            f"{target.item():.6f}"
        )

        # ----------------------------------------------------
        # Backpropagation
        # ----------------------------------------------------

        target.backward()

        if self.activations is None:

            raise RuntimeError(
                "Activations were not captured."
            )

        if self.gradients is None:

            raise RuntimeError(
                "Gradients were not captured."
            )

        activations = (
            self.activations
        )

        gradients = (
            self.gradients
        )

        logger.info(
            f"Activation shape: "
            f"{tuple(activations.shape)}"
        )

        logger.info(
            f"Gradient shape: "
            f"{tuple(gradients.shape)}"
        )

        # ----------------------------------------------------
        # Global average pooling
        # ----------------------------------------------------

        weights = gradients.mean(
            dim=(
                2,
                3,
                4,
            ),
            keepdim=True,
        )

        # ----------------------------------------------------
        # Weighted activations
        # ----------------------------------------------------

        cam = (
            weights *
            activations
        ).sum(
            dim=1,
            keepdim=True,
        )

        # Positive relevance only
        cam = F.relu(
            cam
        )

        # ----------------------------------------------------
        # Resize to 96x96x96
        # ----------------------------------------------------

        cam = F.interpolate(
            cam,
            size=image.shape[2:],
            mode="trilinear",
            align_corners=False,
        )

        cam = cam[
            0,
            0
        ]

        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        cam_min = cam.min()
        cam_max = cam.max()

        cam = (
            cam - cam_min
        ) / (
            cam_max
            - cam_min
            + 1e-8
        )

        return (
            cam.detach()
            .cpu()
            .numpy(),

            probability.detach()
            .cpu()
            .numpy(),
        )


# ============================================================
# TARGET LAYER
# ============================================================

def find_target_layer(
    model,
):
    """
    Select a deep feature-producing convolution.

    The final convolution produces segmentation logits,
    therefore the preceding feature convolution is selected.
    """

    conv_layers = []

    for name, module in (
        model.named_modules()
    ):

        if isinstance(
            module,
            nn.Conv3d,
        ):

            conv_layers.append(
                (
                    name,
                    module,
                )
            )

    if len(conv_layers) < 2:

        raise RuntimeError(
            "Not enough Conv3d layers."
        )

    logger.info("=" * 70)
    logger.info("CONVOLUTIONAL LAYERS")
    logger.info("=" * 70)

    for name, _ in conv_layers:

        logger.info(
            f"{name}"
        )

    # Final convolution is output layer.
    target_name, target_layer = (
        conv_layers[-2]
    )

    logger.info("=" * 70)
    logger.info(
        f"Selected Grad-CAM layer: "
        f"{target_name}"
    )
    logger.info("=" * 70)

    return (
        target_name,
        target_layer,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_dice(
    prediction,
    target,
):

    prediction = (
        prediction.astype(bool)
    )

    target = (
        target.astype(bool)
    )

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    denominator = (
        prediction.sum()
        + target.sum()
    )

    if denominator == 0:

        return 1.0

    return (
        2.0 * intersection
        / denominator
    )


def calculate_iou(
    prediction,
    target,
):

    prediction = (
        prediction.astype(bool)
    )

    target = (
        target.astype(bool)
    )

    intersection = np.logical_and(
        prediction,
        target,
    ).sum()

    union = np.logical_or(
        prediction,
        target,
    ).sum()

    if union == 0:

        return 1.0

    return (
        intersection
        / union
    )


# ============================================================
# FIND BEST SLICES
# ============================================================

def find_best_slices(
    heatmap,
):

    axial = int(
        np.argmax(
            heatmap.sum(
                axis=(
                    1,
                    2,
                )
            )
        )
    )

    coronal = int(
        np.argmax(
            heatmap.sum(
                axis=(
                    0,
                    2,
                )
            )
        )
    )

    sagittal = int(
        np.argmax(
            heatmap.sum(
                axis=(
                    0,
                    1,
                )
            )
        )
    )

    return (
        axial,
        coronal,
        sagittal,
    )


# ============================================================
# SAVE FIGURE
# ============================================================

def save_diagnostic_figure(
    background,
    heatmap,
    prediction,
    ground_truth,
    plane,
    slice_index,
    output_path,
):
    """
    Save:

        Original FLAIR
        Grad-CAM
        Ground Truth
        Prediction
    """

    if plane == "axial":

        bg = background[
            slice_index
        ]

        hm = heatmap[
            slice_index
        ]

        pred = prediction[
            slice_index
        ]

        gt = ground_truth[
            slice_index
        ]

    elif plane == "coronal":

        bg = background[
            :,
            slice_index,
            :
        ]

        hm = heatmap[
            :,
            slice_index,
            :
        ]

        pred = prediction[
            :,
            slice_index,
            :
        ]

        gt = ground_truth[
            :,
            slice_index,
            :
        ]

    else:

        bg = background[
            :,
            :,
            slice_index
        ]

        hm = heatmap[
            :,
            :,
            slice_index
        ]

        pred = prediction[
            :,
            :,
            slice_index
        ]

        gt = ground_truth[
            :,
            :,
            slice_index
        ]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(
            20,
            5,
        ),
    )

    # --------------------------------------------------------
    # FLAIR
    # --------------------------------------------------------

    axes[0].imshow(
        bg,
        cmap="gray",
    )

    axes[0].set_title(
        f"FLAIR - "
        f"{plane.capitalize()}"
    )

    axes[0].axis("off")

    # --------------------------------------------------------
    # GRAD-CAM
    # --------------------------------------------------------

    axes[1].imshow(
        bg,
        cmap="gray",
    )

    axes[1].imshow(
        hm,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    axes[1].set_title(
        "Grad-CAM"
    )

    axes[1].axis("off")

    # --------------------------------------------------------
    # GROUND TRUTH
    # --------------------------------------------------------

    axes[2].imshow(
        bg,
        cmap="gray",
    )

    axes[2].contour(
        gt.astype(
            np.float32
        ),
        levels=[
            0.5
        ],
        linewidths=1.5,
    )

    axes[2].set_title(
        "Ground Truth"
    )

    axes[2].axis("off")

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    axes[3].imshow(
        bg,
        cmap="gray",
    )

    axes[3].contour(
        pred.astype(
            np.float32
        ),
        levels=[
            0.5
        ],
        linewidths=1.5,
    )

    axes[3].set_title(
        "Prediction"
    )

    axes[3].axis("off")

    fig.suptitle(
        f"BraTS 2021 - {plane.capitalize()} "
        f"slice {slice_index}"
    )

    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(
        fig
    )


# ============================================================
# PROCESS PATIENT
# ============================================================

def process_patient(
    model,
    patient,
    transform,
    target_layer_name,
    target_layer,
):

    patient_id = (
        patient["patient_id"]
    )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        f"PROCESSING: {patient_id}"
    )
    logger.info("=" * 70)

    # --------------------------------------------------------
    # Prepare
    # --------------------------------------------------------

    image, ground_truth = (
        prepare_patient(
            patient,
            transform,
        )
    )

    logger.info(
        f"Input shape: "
        f"{tuple(image.shape)}"
    )

    # --------------------------------------------------------
    # Full-volume prediction
    #
    # IMPORTANT:
    # no gradients here
    # --------------------------------------------------------

    image_gpu = image.to(
        DEVICE
    )

    ground_truth_gpu = (
        ground_truth.to(
            DEVICE
        )
    )

    probability = (
        run_full_volume_inference(
            model,
            image_gpu,
        )
    )

    # --------------------------------------------------------
    # Find predicted tumor center
    # --------------------------------------------------------

    tumor_center = (
        find_tumor_center(
            probability
        )
    )

    # --------------------------------------------------------
    # Extract Grad-CAM patch
    # --------------------------------------------------------

    patch, coordinates = (
        extract_centered_patch(
            image_gpu,
            tumor_center,
            PATCH_SIZE,
        )
    )

    patch_gt, _ = (
        extract_centered_patch(
            ground_truth_gpu,
            tumor_center,
            PATCH_SIZE,
        )
    )

    # --------------------------------------------------------
    # Grad-CAM
    # --------------------------------------------------------

    patch.requires_grad_(
        True
    )

    gradcam = GradCAM3D(
        model,
        target_layer,
    )

    try:

        heatmap, patch_probability = (
            gradcam.generate(
                patch
            )
        )

    finally:

        gradcam.remove_hooks()

    # --------------------------------------------------------
    # Patch prediction
    # --------------------------------------------------------

    patch_prediction = (
        patch_probability[
            0,
            0
        ]
        > THRESHOLD
    )

    patch_gt_np = (
        patch_gt[
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    )

    # --------------------------------------------------------
    # Patch metrics
    # --------------------------------------------------------

    patch_dice = (
        calculate_dice(
            patch_prediction,
            patch_gt_np,
        )
    )

    patch_iou = (
        calculate_iou(
            patch_prediction,
            patch_gt_np,
        )
    )

    logger.info(
        f"Patch Dice: "
        f"{patch_dice:.6f}"
    )

    logger.info(
        f"Patch IoU: "
        f"{patch_iou:.6f}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    patient_output = (
        OUTPUT_ROOT /
        patient_id
    )

    patient_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Convert image
    # --------------------------------------------------------

    image_cpu = (
        image_gpu
        .detach()
        .cpu()
    )

    flair_patch = (
        image_cpu[
            0,
            3
        ]
        .numpy()
    )

    # --------------------------------------------------------
    # Full-volume arrays
    # --------------------------------------------------------

    full_probability = (
        probability[
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
    )

    full_prediction = (
        full_probability
        > THRESHOLD
    )

    full_gt = (
        ground_truth[
            0,
            0
        ]
        .detach()
        .cpu()
        .numpy()
        > 0.5
    )

    # --------------------------------------------------------
    # Full-volume metrics
    # --------------------------------------------------------

    full_dice = calculate_dice(
        full_prediction,
        full_gt,
    )

    full_iou = calculate_iou(
        full_prediction,
        full_gt,
    )

    logger.info(
        f"Full-volume Dice: "
        f"{full_dice:.6f}"
    )

    logger.info(
        f"Full-volume IoU: "
        f"{full_iou:.6f}"
    )

    # --------------------------------------------------------
    # Place heatmap in full-volume coordinates
    # --------------------------------------------------------

    full_heatmap = (
        place_patch_in_volume(
            heatmap,
            full_gt.shape,
            coordinates,
        )
    )

    # --------------------------------------------------------
    # Best slices inside patch
    # --------------------------------------------------------

    axial, coronal, sagittal = (
        find_best_slices(
            heatmap
        )
    )

    # --------------------------------------------------------
    # Save diagnostic figures
    # --------------------------------------------------------

    save_diagnostic_figure(
        flair_patch,
        heatmap,
        patch_prediction,
        patch_gt_np,
        "axial",
        axial,
        patient_output /
        "axial_diagnostic.png",
    )

    save_diagnostic_figure(
        flair_patch,
        heatmap,
        patch_prediction,
        patch_gt_np,
        "coronal",
        coronal,
        patient_output /
        "coronal_diagnostic.png",
    )

    save_diagnostic_figure(
        flair_patch,
        heatmap,
        patch_prediction,
        patch_gt_np,
        "sagittal",
        sagittal,
        patient_output /
        "sagittal_diagnostic.png",
    )

    # --------------------------------------------------------
    # Save raw data
    # --------------------------------------------------------

    np.save(
        patient_output /
        "gradcam.npy",
        full_heatmap,
    )

    np.save(
        patient_output /
        "patch_gradcam.npy",
        heatmap,
    )

    np.save(
        patient_output /
        "patch_prediction.npy",
        patch_prediction.astype(
            np.uint8
        ),
    )

    np.save(
        patient_output /
        "full_volume_probability.npy",
        full_probability,
    )

    np.save(
        patient_output /
        "ground_truth.npy",
        full_gt.astype(
            np.uint8
        ),
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    with open(
        patient_output /
        "metadata.txt",
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            f"Patient ID: {patient_id}\n"
        )

        file.write(
            f"Checkpoint: {CHECKPOINT_PATH}\n"
        )

        file.write(
            "Checkpoint Epoch: 25\n"
        )

        file.write(
            "Model: MONAI SegResNet\n"
        )

        file.write(
            "Input channels: T1, T1ce, T2, FLAIR\n"
        )

        file.write(
            f"Grad-CAM target layer: "
            f"{target_layer_name}\n"
        )

        file.write(
            f"Original volume shape: "
            f"{full_gt.shape}\n"
        )

        file.write(
            f"Grad-CAM patch shape: "
            f"{heatmap.shape}\n"
        )

        file.write(
            f"Tumor center: "
            f"{tuple(tumor_center.tolist())}\n"
        )

        file.write(
            f"Patch start coordinates: "
            f"{coordinates}\n"
        )

        file.write(
            f"Threshold: {THRESHOLD}\n"
        )

        file.write(
            f"Full-volume Dice: "
            f"{full_dice:.6f}\n"
        )

        file.write(
            f"Full-volume IoU: "
            f"{full_iou:.6f}\n"
        )

        file.write(
            f"Patch Dice: "
            f"{patch_dice:.6f}\n"
        )

        file.write(
            f"Patch IoU: "
            f"{patch_iou:.6f}\n"
        )

        file.write(
            f"Axial patch slice: "
            f"{axial}\n"
        )

        file.write(
            f"Coronal patch slice: "
            f"{coronal}\n"
        )

        file.write(
            f"Sagittal patch slice: "
            f"{sagittal}\n"
        )

    # --------------------------------------------------------
    # Free GPU memory
    # --------------------------------------------------------

    del (
        image_gpu,
        ground_truth_gpu,
        probability,
        patch,
        patch_gt,
    )

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    logger.info("=" * 70)
    logger.info(
        f"RESULTS SAVED:\n"
        f"{patient_output}"
    )
    logger.info("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "BRATS 2021 - MEMORY-EFFICIENT 3D GRAD-CAM"
    )
    logger.info("=" * 70)

    logger.info(
        f"Device: {DEVICE}"
    )

    if torch.cuda.is_available():

        logger.info(
            f"GPU: "
            f"{torch.cuda.get_device_name(0)}"
        )

        logger.info(
            f"VRAM: "
            f"{torch.cuda.get_device_properties(0).total_memory / (1024 ** 3):.2f} GB"
        )

    logger.info(
        f"Patch size: "
        f"{PATCH_SIZE}"
    )

    logger.info(
        f"Sliding-window batch: "
        f"{SW_BATCH_SIZE}"
    )

    logger.info(
        f"Checkpoint:\n"
        f"{CHECKPOINT_PATH}"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model()

    # --------------------------------------------------------
    # Target layer
    # --------------------------------------------------------

    (
        target_layer_name,
        target_layer,
    ) = find_target_layer(
        model
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    transform = (
        create_base_transform()
    )

    # --------------------------------------------------------
    # Patients
    # --------------------------------------------------------

    patients = (
        find_patients()
    )

    if len(patients) == 0:

        raise RuntimeError(
            "No patients found."
        )

    patients = patients[
        :NUM_PATIENTS
    ]

    logger.info(
        f"Selected patients: "
        f"{len(patients)}"
    )

    # --------------------------------------------------------
    # Process
    # --------------------------------------------------------

    for patient in patients:

        process_patient(
            model,
            patient,
            transform,
            target_layer_name,
            target_layer,
        )

    logger.info("")
    logger.info("=" * 70)
    logger.info(
        "GRAD-CAM COMPLETE"
    )
    logger.info("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()