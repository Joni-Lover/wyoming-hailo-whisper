"""Tests for backend selection in the command-line entry point."""

import asyncio
import builtins
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock


def test_cpu_mode_does_not_import_hailo(monkeypatch):
    """CPU mode starts without importing the optional Hailo runtime."""
    main_module_name = "wyoming_hailo_whisper.__main__"
    hailo_pipeline_name = "wyoming_hailo_whisper.app.hailo_whisper_pipeline"
    cpu_pipeline_name = "wyoming_hailo_whisper.app.cpu_whisper_pipeline"

    monkeypatch.delitem(sys.modules, main_module_name, raising=False)
    monkeypatch.delitem(sys.modules, hailo_pipeline_name, raising=False)

    original_import = builtins.__import__

    def reject_hailo_import(name, *args, **kwargs):
        if name == "hailo_platform" or name == hailo_pipeline_name:
            raise AssertionError(f"CPU mode imported optional Hailo module: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_hailo_import)

    fake_model = MagicMock()
    fake_cpu_module = ModuleType(cpu_pipeline_name)
    fake_cpu_module.CpuWhisperPipeline = MagicMock(return_value=fake_model)
    monkeypatch.setitem(sys.modules, cpu_pipeline_name, fake_cpu_module)

    main_module = importlib.import_module(main_module_name)
    fake_server = MagicMock()
    fake_server.run = MagicMock(return_value=asyncio.sleep(0))
    monkeypatch.setattr(
        main_module.AsyncServer,
        "from_uri",
        MagicMock(return_value=fake_server),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["wyoming_hailo_whisper", "--uri", "tcp://127.0.0.1:10600", "--use-cpu"],
    )

    asyncio.run(main_module.main())

    fake_cpu_module.CpuWhisperPipeline.assert_called_once()
    fake_model.stop.assert_called_once()
