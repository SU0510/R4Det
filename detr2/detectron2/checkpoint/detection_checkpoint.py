# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Any, Dict, Optional
import torch.nn as nn

logger = logging.getLogger(__name__)


class DetectionCheckpointer:
    """
    Minimal stub of Detectron2's DetectionCheckpointer.
    
    Only satisfies import requirements for model_zoo. If you need
    actual checkpoint loading (model_zoo.get()), install full detectron2:
    
        pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html
    
    This stub is safe for gen_panoptic_seg_TJ4D.py which only uses
    model_zoo.get_config_file() and model_zoo.get_checkpoint_url() —
    neither of which instantiates DetectionCheckpointer.
    """

    def __init__(self, model: nn.Module, save_dir: str = "", **kwargs):
        self.model = model
        self.save_dir = save_dir
        self.checkpointables = {}
        logger.debug(
            "DetectionCheckpointer stub initialized. "
            "Full checkpoint I/O disabled."
        )

    def load(self, path: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError(
            "DetectionCheckpointer.load() called, but only a stub is installed. "
            "Install full detectron2 for checkpoint loading: "
            "pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu118/torch2.0/index.html"
        )

    def save(self, name: str, **kwargs) -> None:
        raise NotImplementedError(
            "DetectionCheckpointer.save() called, but only a stub is installed."
        )

    def resume_or_load(self, path: str, *, resume: bool = True) -> Dict[str, Any]:
        return self.load(path)

    def has_checkpoint(self) -> bool:
        return False

    def get_checkpoint_file(self) -> str:
        return ""

    def get_all_checkpoint_files(self):
        return []

    def tag_last_checkpoint(self, last_filename_basename: str) -> None:
        pass
