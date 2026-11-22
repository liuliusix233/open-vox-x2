#!/usr/bin/env python3
"""vox_v2 entrypoint: binds local SenseVoice ASR + CosyVoice2 TTS, zero regex routers."""

from __future__ import annotations

import logging
import os

from aiohttp import web

from vox_v2.app import create_app
from vox_v2.asr_local import sensevoice_asr
from vox_v2.config import VoxConfig


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = VoxConfig.from_env()
    app = create_app(cfg, asr_override=sensevoice_asr)
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
