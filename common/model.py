"""
Manages the storage and utility of model containers.

Containers exist as a common interface for backends.
"""

import asyncio
import gc
import pathlib
from enum import Enum
from fastapi import HTTPException
from loguru import logger
from typing import Optional
import torch

from common.logger import get_loading_progress_bar
from common.networking import handle_request_error
from common.tabby_config import config
from common.optional_dependencies import dependencies

if dependencies.exllamav2:
    from backends.exllamav2.model import ExllamaV2Container

    # Global model container
    container: Optional[ExllamaV2Container] = None
    embeddings_container = None


if dependencies.extras:
    from backends.infinity.model import InfinityContainer

    embeddings_container: Optional[InfinityContainer] = None


class ModelType(Enum):
    MODEL = "model"
    DRAFT = "draft"
    EMBEDDING = "embedding"
    VISION = "vision"


def load_progress(module, modules):
    """Wrapper callback for load progress."""
    yield module, modules


async def unload_model(skip_wait: bool = False, shutdown: bool = False):
    """Unloads a model"""
    global container

    await container.unload(skip_wait=skip_wait, shutdown=shutdown)
    container = None


async def load_model_gen(model_path: pathlib.Path, **kwargs):
    """Generator to load a model"""
    global container

    # Check if the model is already loaded
    if container and container.model:
        loaded_model_name = container.model_dir.name

        if loaded_model_name == model_path.name and container.model_loaded:
            raise ValueError(
                f'Model "{loaded_model_name}" is already loaded! Aborting.'
            )

        # Remove API request flag and other parameters that unload_model doesn't accept
        unload_kwargs = {k: v for k, v in kwargs.items() if k in ['skip_wait', 'shutdown']}
        
        logger.info("Unloading existing model.")
        await unload_model(**unload_kwargs)
        
        # The unload method now has a built-in delay to ensure resources are freed
        logger.info("Preparing to load new model...")

    # Merge with config defaults
    kwargs = {**config.model_defaults, **kwargs}

    # Add an API request flag to determine if progress bar should be shown
    is_api_request = kwargs.pop("is_api_request", False)

    # Create a new container
    container = await ExllamaV2Container.create(model_path.resolve(), False, **kwargs)

    # Add possible types of models that can be loaded
    model_type = [ModelType.MODEL]

    if container.use_vision:
        model_type.insert(0, ModelType.VISION)

    if container.draft_config:
        model_type.insert(0, ModelType.DRAFT)

    # Pass the API request flag to load_gen
    load_status = container.load_gen(load_progress, is_api_request=is_api_request, **kwargs)

    progress = None
    try:
        # Only create progress bar for non-API requests
        if not is_api_request:
            progress = get_loading_progress_bar()
            progress.start()

        index = 0
        async for module, modules in load_status:
            current_model_type = model_type[index].value
            if module == 0:
                if progress:  # Only update progress if it exists
                    loading_task = progress.add_task(
                        f"[cyan]Loading {current_model_type} modules", total=modules
                    )
            else:
                if progress:  # Only update progress if it exists
                    progress.advance(loading_task)

            yield module, modules, current_model_type

            if module == modules:
                # Switch to model progress if the draft model is loaded
                if index == len(model_type) - 1:
                    pass  # Progress will be stopped in finally
                else:
                    index += 1
    except Exception as e:
        logger.error(f"Error during model loading: {str(e)}")
        if container:
            try:
                await container.unload(skip_wait=True)
            except Exception as unload_error:
                logger.error(f"Error during cleanup: {str(unload_error)}")
        raise
    finally:
        if progress:
            progress.stop()

async def load_model(model_path: pathlib.Path, **kwargs):
    """Wrapper to load a model synchronously."""
    async for _ in load_model_gen(model_path, **kwargs):
        pass


async def load_loras(lora_dir, **kwargs):
    """Wrapper to load loras."""
    if len(container.get_loras()) > 0:
        await unload_loras()

    return await container.load_loras(lora_dir, **kwargs)


async def unload_loras():
    """Wrapper to unload loras"""
    await container.unload(loras_only=True)


async def load_embedding_model(model_path: pathlib.Path, **kwargs):
    global embeddings_container

    # Break out if infinity isn't installed
    if not dependencies.extras:
        raise ImportError(
            "Skipping embeddings because infinity-emb is not installed.\n"
            "Please run the following command in your environment "
            "to install extra packages:\n"
            "pip install -U .[extras]"
        )

    # Check if the model is already loaded
    if embeddings_container and embeddings_container.engine:
        loaded_model_name = embeddings_container.model_dir.name

        if loaded_model_name == model_path.name and embeddings_container.model_loaded:
            raise ValueError(
                f'Embeddings model "{loaded_model_name}" is already loaded! Aborting.'
            )

        logger.info("Unloading existing embeddings model.")
        await unload_embedding_model()

    embeddings_container = InfinityContainer(model_path)
    await embeddings_container.load(**kwargs)


async def unload_embedding_model():
    global embeddings_container

    await embeddings_container.unload()
    embeddings_container = None


async def check_model_container():
    """FastAPI depends that checks if a model isn't loaded or currently loading."""

    if container is None or not (container.model_is_loading or container.model_loaded):
        error_message = handle_request_error(
            "No models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)


async def check_embeddings_container():
    """
    FastAPI depends that checks if an embeddings model is loaded.

    This is the same as the model container check, but with embeddings instead.
    """

    if embeddings_container is None or not (
        embeddings_container.model_is_loading or embeddings_container.model_loaded
    ):
        error_message = handle_request_error(
            "No embedding models are currently loaded.",
            exc_info=False,
        ).error.message

        raise HTTPException(400, error_message)

# Function removed - relying on ExLlamaV2's unload methods instead