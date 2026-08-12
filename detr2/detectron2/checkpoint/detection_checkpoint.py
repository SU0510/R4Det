# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import os
import tempfile
from typing import Any, Dict, Optional, List
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class DetectionCheckpointer:
    """
    Detectron2-compatible checkpointer that supports loading from URLs,
    local files, and detectron2:// protocol paths.

    Handles the common detectron2 checkpoint formats:
    - Plain state_dict
    - Dict with 'model' key containing state_dict
    - Dict with 'state_dict' key
    """

    def __init__(self, model: nn.Module, save_dir: str = "", **kwargs):
        self.model = model
        self.save_dir = save_dir
        self.checkpointables = {}

    def _download_and_load(self, path: str) -> Dict[str, Any]:
        """
        Download a checkpoint from a URL and load it with torch.load.
        Handles detectron2 .pkl format (pickle).
        Uses /tmp for caching to avoid read-only filesystem issues.
        """
        import urllib.request
        import hashlib

        # Use /tmp for download cache
        cache_dir = os.path.join(tempfile.gettempdir(), 'detectron2_checkpoints')
        os.makedirs(cache_dir, exist_ok=True)

        # Create a filename from the URL hash
        url_hash = hashlib.md5(path.encode()).hexdigest()[:12]
        ext = '.pkl' if path.endswith('.pkl') else '.pth'
        local_path = os.path.join(cache_dir, f'{url_hash}{ext}')

        if not os.path.exists(local_path):
            logger.info(f"Downloading checkpoint from {path} ...")
            logger.info(f"Saving to {local_path}")
            try:
                urllib.request.urlretrieve(path, local_path)
            except Exception as e:
                logger.error(f"Download failed: {e}")
                raise
        else:
            logger.info(f"Using cached checkpoint: {local_path}")

        checkpoint = torch.load(local_path, map_location=torch.device('cpu'))
        return checkpoint

    def _load_from_file(self, path: str) -> Dict[str, Any]:
        """Load a local checkpoint file."""
        checkpoint = torch.load(path, map_location=torch.device('cpu'))
        return checkpoint

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract the model state dict from various checkpoint formats."""
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                return checkpoint['model']
            if 'state_dict' in checkpoint:
                return checkpoint['state_dict']
        return checkpoint

    def load(self, path: str, **kwargs) -> Dict[str, Any]:
        """
        Load checkpoint from a URL or local path.

        Supports:
        - HTTP/HTTPS URLs
        - detectron2:// protocol (maps to fbaipublicfiles)
        - Local file paths
        """
        import re

        if not path or path == '':
            logger.warning("No checkpoint path provided, skipping load.")
            return {}

        # Map detectron2:// to https://
        if path.startswith('detectron2://'):
            path = re.sub(r'^detectron2://', 'https://dl.fbaipublicfiles.com/detectron2/', path)

        logger.info(f"Loading checkpoint from: {path}")

        # Determine if URL or local file
        if path.startswith(('http://', 'https://')):
            checkpoint = self._download_and_load(path)
        else:
            if not os.path.exists(path):
                logger.warning(f"Checkpoint file not found: {path}, skipping.")
                return {}
            checkpoint = self._load_from_file(path)

        state_dict = self._extract_state_dict(checkpoint)

        # Handle DataParallel/model wrapper prefixes
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

        # Load into model
        incompatible = self.model.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            logger.info(f"Missing keys: {len(incompatible.missing_keys)}")
        if incompatible.unexpected_keys:
            logger.info(f"Unexpected keys: {len(incompatible.unexpected_keys)}")

        logger.info(f"Checkpoint loaded successfully from {path}")
        return checkpoint

    def save(self, name: str, **kwargs) -> None:
        if not self.save_dir:
            self.save_dir = '.'
        save_path = os.path.join(self.save_dir, name)
        checkpoint = {
            'model': self.model.state_dict(),
        }
        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        return self.load(path)

    def has_checkpoint(self) -> bool:
        return False

    def get_checkpoint_file(self) -> str:
        return ""

    def get_all_checkpoint_files(self) -> List[str]:
        return []

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        pass
