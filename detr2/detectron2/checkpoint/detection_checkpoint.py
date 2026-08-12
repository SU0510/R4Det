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
    """

    def __init__(self, model: nn.Module, save_dir: str = "", **kwargs):
        self.model = model
        self.save_dir = save_dir
        self.checkpointables = {}

    @staticmethod
    def _load_checkpoint(filepath):
        """Load checkpoint file. Uses pickle for .pkl (detectron2 format), torch for .pth."""
        if filepath.endswith('.pkl'):
            import pickle
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        else:
            return torch.load(filepath, map_location=torch.device('cpu'))


    def _download_and_load(self, path: str) -> Dict[str, Any]:
        import subprocess
        import hashlib

        cache_dir = os.path.join(tempfile.gettempdir(), 'detectron2_checkpoints')
        os.makedirs(cache_dir, exist_ok=True)

        url_hash = hashlib.md5(path.encode()).hexdigest()[:12]
        ext = '.pkl' if path.endswith('.pkl') else '.pth'
        local_path = os.path.join(cache_dir, f'{url_hash}{ext}')

        # Also check for file saved with original URL basename (user manual download)
        orig_basename = os.path.basename(path)
        alt_path = os.path.join(cache_dir, orig_basename)
        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 100 * 1024 * 1024:
            if not os.path.exists(local_path) or os.path.getsize(alt_path) > os.path.getsize(local_path):
                logger.info(f"Found alternative cache file: {alt_path}, using it")
                if os.path.exists(local_path):
                    os.remove(local_path)
                os.rename(alt_path, local_path)

        # Validate cached file (check header bytes first to avoid loading corrupted files)
        if os.path.exists(local_path):
            with open(local_path, 'rb') as f:
                header = f.read(8)
            valid_headers = (b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05', b'PK')
            if header[:2] in valid_headers:
                logger.info(f"Using cached checkpoint: {local_path} ({os.path.getsize(local_path)/1024/1024:.0f} MB)")
                return self._load_checkpoint(local_path)
            else:
                logger.warning(
                    f"Cached file has invalid header ({header[:4].hex()}), may be HTML/corrupt. "
                    f"Will re-download but keeping old file as backup."
                )
                # Don't delete — keep as backup in case re-download also fails
                os.rename(local_path, local_path + '.bak')

        # Download
        logger.info(f"Downloading checkpoint from {path} ...")
        tmp_path = local_path + '.tmp'
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

        download_ok = False
        last_error = None

        # Strategy 1: without proxy
        for strategy, extra_args in [
            ("direct (no proxy)", ['--noproxy', '*']),
            ("with system proxy", []),
        ]:
            try:
                logger.info(f"Trying download {strategy}...")
                cmd = ['curl', '-sL', '--retry', '3', '--retry-delay', '5',
                       '--connect-timeout', '60', '--max-time', '1800',
                       '-o', tmp_path, path] + extra_args
                result = subprocess.run(cmd, capture_output=False, text=True)

                if result.returncode != 0:
                    raise RuntimeError(f"curl exit code {result.returncode}")
                if not os.path.exists(tmp_path):
                    raise RuntimeError("no output file")

                file_size = os.path.getsize(tmp_path)
                logger.info(f"Downloaded {file_size / 1024 / 1024:.0f} MB, validating...")

                # Check header bytes
                with open(tmp_path, 'rb') as f:
                    header = f.read(8)
                valid_headers = (b'\x80\x02', b'\x80\x03', b'\x80\x04', b'\x80\x05', b'PK')
                if header[:2] not in valid_headers:
                    raise RuntimeError(
                        "Invalid file format. First 8 bytes: {} ({}). "
                        "File may be HTML, not a pickle checkpoint.".format(
                            header.hex(), repr(header[:60])
                        )
                    )

                # Full validation
                self._load_checkpoint(tmp_path)
                os.rename(tmp_path, local_path)
                logger.info(f"Saved to {local_path}")
                download_ok = True
                break
            except Exception as e:
                last_error = e
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                logger.warning(f"Download {strategy} failed: {e}")
                continue

        if not download_ok:
            msg = (
                "All download strategies failed. Last error: {}. ".format(last_error) +
                "Please download manually:\n"
                "  curl -L -o {} '{}'\n".format(local_path, path) +
                "Then re-run this script."
            )
            raise RuntimeError(msg)

        return self._load_checkpoint(local_path)

    def _extract_state_dict(self, checkpoint: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model state dict and convert numpy arrays to torch tensors."""
        import numpy as np
        from collections import OrderedDict

        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Convert OrderedDict to regular dict
        if isinstance(state_dict, OrderedDict):
            state_dict = dict(state_dict)

        # Convert numpy arrays to torch tensors
        converted = {}
        for k, v in state_dict.items():
            if isinstance(v, np.ndarray):
                converted[k] = torch.from_numpy(v)
            elif isinstance(v, OrderedDict):
                # Nested OrderedDict (e.g., from collections.OrderedDict)
                converted[k] = {sk: torch.from_numpy(sv) if isinstance(sv, np.ndarray) else sv
                               for sk, sv in v.items()}
            else:
                converted[k] = v

        return converted

    def load(self, path: str, **kwargs) -> Dict[str, Any]:
        import re

        if not path or path == '':
            logger.warning("No checkpoint path provided, skipping load.")
            return {}

        if path.startswith('detectron2://'):
            path = re.sub(r'^detectron2://', 'https://dl.fbaipublicfiles.com/detectron2/', path)

        logger.info(f"Loading checkpoint from: {path}")

        if path.startswith(('http://', 'https://')):
            checkpoint = self._download_and_load(path)
        else:
            if not os.path.exists(path):
                logger.warning(f"Checkpoint file not found: {path}, skipping.")
                return {}
            checkpoint = self._load_checkpoint(path)

        state_dict = self._extract_state_dict(checkpoint)
        if any(k.startswith('module.') for k in state_dict.keys()):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}

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
        checkpoint = {'model': self.model.state_dict()}
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
