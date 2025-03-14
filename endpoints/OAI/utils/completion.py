"""
Completion utilities for OAI server.

Also serves as a common module for completions and chat completions.
"""

import asyncio
import pathlib
from asyncio import CancelledError
from fastapi import HTTPException, Request
from typing import List, Union

from loguru import logger

from common import model
from common.auth import get_key_permission
from common.networking import (
    get_generator_error,
    handle_request_disconnect,
    handle_request_error,
    request_disconnect_loop,
)
from common.tabby_config import config
from common.utils import unwrap
from endpoints.OAI.types.completion import (
    CompletionRequest,
    CompletionResponse,
    CompletionRespChoice,
    CompletionLogProbs,
)
from endpoints.OAI.types.common import UsageStats


def _create_response(
    request_id: str, generations: Union[dict, List[dict]], model_name: str = ""
):
    """Create a completion response from the provided choices."""

    # Convert the single choice object into a list
    if not isinstance(generations, list):
        generations = [generations]

    choices: List[CompletionRespChoice] = []
    for index, generation in enumerate(generations):
        logprob_response = None

        token_probs = unwrap(generation.get("token_probs"), {})
        if token_probs:
            logprobs = unwrap(generation.get("logprobs"), [])
            offset = unwrap(generation.get("offset"), [])

            logprob_response = CompletionLogProbs(
                text_offset=offset if isinstance(offset, list) else [offset],
                token_logprobs=token_probs.values(),
                tokens=token_probs.keys(),
                top_logprobs=logprobs if isinstance(logprobs, list) else [logprobs],
            )

        # The index can be located in the generation itself
        choice = CompletionRespChoice(
            index=unwrap(generation.get("index"), index),
            finish_reason=generation.get("finish_reason"),
            text=unwrap(generation.get("text"), ""),
            logprobs=logprob_response,
        )

        choices.append(choice)

    prompt_tokens = unwrap(generations[-1].get("prompt_tokens"), 0)
    completion_tokens = unwrap(generations[-1].get("generated_tokens"), 0)

    response = CompletionResponse(
        id=f"cmpl-{request_id}",
        choices=choices,
        model=model_name,
        usage=UsageStats(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )

    return response


async def _stream_collector(
    task_idx: int,
    gen_queue: asyncio.Queue,
    prompt: str,
    request_id: str,
    abort_event: asyncio.Event,
    **kwargs,
):
    """Collects a stream and places results in a common queue"""

    try:
        new_generation = model.container.generate_gen(
            prompt, request_id, abort_event, **kwargs
        )
        async for generation in new_generation:
            generation["index"] = task_idx

            await gen_queue.put(generation)

            if "finish_reason" in generation:
                break
    except Exception as e:
        await gen_queue.put(e)


async def load_inline_model(model_name: str, request: Request):
    """Load a model from the data.model parameter"""

    # Return if the model container already exists and the model is fully loaded
    if (
        model.container
        and model.container.model_dir.name == model_name
        and model.container.model_loaded
    ):
        return

    # Return if inline loading is disabled
    # Also warn if an admin key is used
    if not config.model.inline_model_loading:
        if get_key_permission(request) == "admin":
            logger.warning(
                f"Unable to switch model to {model_name} because "
                '"inline_model_loading" is not True in config.yml.'
            )

        return

    is_dummy_model = (
        config.model.use_dummy_models and model_name in config.model.dummy_model_names
    )

    # Error if an invalid key is passed
    # If a dummy model is provided, don't error
    if get_key_permission(request) != "admin":
        if not is_dummy_model:
            error_message = handle_request_error(
                f"Unable to switch model to {model_name} because "
                + "an admin key isn't provided",
                exc_info=False,
            ).error.message

            raise HTTPException(401, error_message)
        else:
            return

    # Skip if the model is a dummy
    if is_dummy_model:
        logger.warning(f"Dummy model {model_name} provided. Skipping inline load.")
        return

    # Check if a model is already loading
    if model.container and model.container.model_is_loading:
        current_model_name = "unknown"
        if model.container.model_dir:
            current_model_name = model.container.model_dir.name
            
        error_message = handle_request_error(
            f"Cannot switch to model '{model_name}' because model '{current_model_name}' is currently loading. "
            "Please try again later.",
            exc_info=False,
        ).error.message
        raise HTTPException(503, error_message)

    # Check if active generations are in progress
    active_jobs = {}
    if model.container and model.container.generator and hasattr(model.container.generator, "jobs"):
        active_jobs = model.container.generator.jobs
        
    if active_jobs:
        current_model_name = model.container.model_dir.name
        job_count = len(active_jobs)
        
        logger.info(f"Model switch request detected {job_count} active generation jobs")
        
        # Check if we should queue or reject based on config setting
        if not config.model.queue_model_switch_requests:
            # Reject with error
            error_message = handle_request_error(
                f"Cannot switch models while active generations are in progress with model '{current_model_name}'. "
                f"Requested model '{model_name}' will not be loaded. Please try again later.",
                exc_info=False,
            ).error.message
            raise HTTPException(503, error_message)
        else:
            # Wait for active jobs WITHOUT holding the load_lock
            logger.info(
                f"Request to switch from '{current_model_name}' to '{model_name}' is waiting for "
                f"{job_count} active generations to complete."
            )
            
            # Create a copy of active job IDs
            active_job_ids = list(active_jobs.keys())
            
            # Wait for all active jobs to complete (with timeout protection)
            max_wait_time = 300  # 5 minutes max wait
            start_time = asyncio.get_event_loop().time()
            
            while active_job_ids and (asyncio.get_event_loop().time() - start_time < max_wait_time):
                # Get current jobs
                current_jobs = getattr(model.container.generator, "jobs", {})
                # Update active job IDs by checking which are still in the generator's jobs dict
                active_job_ids = [job_id for job_id in active_job_ids if job_id in current_jobs]
                
                if not active_job_ids:
                    logger.info(f"All active generations completed. Proceeding with model switch to '{model_name}'.")
                    # Add a small delay to ensure generators are fully closed before unloading
                    await asyncio.sleep(0.5)
                    break
                    
                # Add periodic logging
                if (asyncio.get_event_loop().time() - start_time) % 10 < 0.1:  # Log every ~10 seconds
                    logger.info(f"Still waiting for {len(active_job_ids)} generation jobs to complete...")
                    
                await asyncio.sleep(0.1)  # Short sleep to prevent CPU spinning
                
            # If we timed out
            if active_job_ids:
                error_message = handle_request_error(
                    f"Timed out waiting for active generations to complete. "
                    f"Cannot switch from '{current_model_name}' to '{model_name}'.",
                    exc_info=False,
                ).error.message
                raise HTTPException(504, error_message)  # 504 Gateway Timeout

    # Force a re-check of active jobs to ensure nothing started while we were waiting
    if hasattr(model.container, "generator") and hasattr(model.container.generator, "jobs"):
        active_jobs = model.container.generator.jobs
        if active_jobs:
            job_count = len(active_jobs)
            current_model_name = model.container.model_dir.name
            error_message = handle_request_error(
                f"New generations started while waiting for switch. "
                f"Cannot switch from '{current_model_name}' to '{model_name}' "
                f"with {job_count} active jobs.",
                exc_info=False,
            ).error.message
            raise HTTPException(503, error_message)
    
    model_path = pathlib.Path(config.model.model_dir)
    model_path = model_path / model_name

    # Model path doesn't exist
    if not model_path.exists():
        logger.warning(
            f"Could not find model path {str(model_path)}. Skipping inline model load."
        )
        return

    # Load the model and also add draft dir with API request flag
    try:
        logger.info(f"Starting model switch to '{model_name}'")
        await model.load_model(
            model_path,
            is_api_request=True,  # Flag this as an API request to skip the progress bar
            draft_model=config.draft_model.model_dump(include={"draft_model_dir"}),
        )
        logger.info(f"Successfully switched to model '{model_name}'")
    except Exception as e:
        error_message = handle_request_error(
            f"Failed to load model '{model_name}': {str(e)}",
            exc_info=True,
        ).error.message
        raise HTTPException(503, error_message)

async def stream_generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Streaming generation for completions."""

    abort_event = asyncio.Event()
    gen_queue = asyncio.Queue()
    gen_tasks: List[asyncio.Task] = []
    disconnect_task = asyncio.create_task(request_disconnect_loop(request))

    try:
        logger.info(f"Received streaming completion request {request.state.id}")

        for n in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)

            gen_task = asyncio.create_task(
                _stream_collector(
                    n,
                    gen_queue,
                    data.prompt,
                    request.state.id,
                    abort_event,
                    **task_gen_params.model_dump(exclude={"prompt"}),
                )
            )

            gen_tasks.append(gen_task)

        # Consumer loop
        while True:
            if disconnect_task.done():
                abort_event.set()
                handle_request_disconnect(
                    f"Completion generation {request.state.id} cancelled by user."
                )

            generation = await gen_queue.get()

            # Stream collector will push an exception to the queue if it fails
            if isinstance(generation, Exception):
                raise generation

            response = _create_response(request.state.id, generation, model_path.name)
            yield response.model_dump_json()

            # Check if all tasks are completed
            if all(task.done() for task in gen_tasks) and gen_queue.empty():
                yield "[DONE]"
                logger.info(f"Finished streaming completion request {request.state.id}")
                break
    except CancelledError:
        # Get out if the request gets disconnected

        if not disconnect_task.done():
            abort_event.set()
            handle_request_disconnect(
                f"Completion generation {request.state.id} cancelled by user."
            )
    except Exception:
        yield get_generator_error(
            f"Completion {request.state.id} aborted. Please check the server console."
        )


async def generate_completion(
    data: CompletionRequest, request: Request, model_path: pathlib.Path
):
    """Non-streaming generate for completions"""

    gen_tasks: List[asyncio.Task] = []

    try:
        logger.info(f"Recieved completion request {request.state.id}")

        for _ in range(0, data.n):
            task_gen_params = data.model_copy(deep=True)

            gen_tasks.append(
                asyncio.create_task(
                    model.container.generate(
                        data.prompt,
                        request.state.id,
                        **task_gen_params.model_dump(exclude={"prompt"}),
                    )
                )
            )

        generations = await asyncio.gather(*gen_tasks)
        response = _create_response(request.state.id, generations, model_path.name)

        logger.info(f"Finished completion request {request.state.id}")

        return response
    except Exception as exc:
        error_message = handle_request_error(
            f"Completion {request.state.id} aborted. Maybe the model was unloaded? "
            "Please check the server console."
        ).error.message

        # Server error if there's a generation exception
        raise HTTPException(503, error_message) from exc
