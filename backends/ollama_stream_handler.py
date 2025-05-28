# backends/ollama_stream_handler.py
import asyncio
import logging
import time
from typing import Optional, AsyncGenerator, Any, Dict
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class OllamaStreamTimeoutError(Exception):
    """Custom exception for Ollama streaming timeouts."""
    pass


class OllamaStreamHandler:
    """
    Dedicated handler for Ollama streaming with robust timeout and retry logic.
    Addresses the ReadTimeout issues seen with Ollama's streaming responses.
    """

    def __init__(self,
                 chunk_timeout: float = 30.0,
                 total_timeout: float = 300.0,  # 5 minutes max per request
                 max_retries: int = 2,
                 retry_delay: float = 2.0):
        self.chunk_timeout = chunk_timeout
        self.total_timeout = total_timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Track active streams for debugging
        self._active_streams: Dict[str, Dict[str, Any]] = {}

    async def stream_with_timeout(self,
                                  ollama_client,
                                  model_name: str,
                                  messages: list,
                                  options: Optional[Dict[str, Any]] = None,
                                  request_id: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream from Ollama with comprehensive timeout handling and retries.

        Args:
            ollama_client: The Ollama client instance
            model_name: Model name to use
            messages: Messages for the chat
            options: Optional parameters
            request_id: Optional request ID for tracking

        Yields:
            String chunks from the model response

        Raises:
            OllamaStreamTimeoutError: If streaming times out beyond recovery
        """

        if not request_id:
            request_id = f"stream_{int(time.time())}"

        # Track this stream
        self._active_streams[request_id] = {
            'start_time': time.time(),
            'last_chunk_time': time.time(),
            'chunk_count': 0,
            'total_chars': 0,
            'model': model_name
        }

        try:
            retry_count = 0
            last_error = None

            while retry_count <= self.max_retries:
                try:
                    logger.info(f"Starting Ollama stream for {model_name} (attempt {retry_count + 1})")

                    async for chunk in self._stream_attempt(
                            ollama_client, model_name, messages, options, request_id
                    ):
                        yield chunk

                    # If we get here, streaming completed successfully
                    logger.info(f"Ollama stream completed successfully for {request_id}")
                    return

                except (asyncio.TimeoutError, OllamaStreamTimeoutError) as e:
                    last_error = e
                    retry_count += 1
                    logger.warning(f"Ollama stream timeout (attempt {retry_count}/{self.max_retries + 1}): {e}")

                    if retry_count <= self.max_retries:
                        logger.info(f"Retrying Ollama stream in {self.retry_delay}s...")
                        await asyncio.sleep(self.retry_delay)
                    else:
                        logger.error(f"All retry attempts exhausted for Ollama stream {request_id}")

                except Exception as e:
                    logger.error(f"Non-timeout error in Ollama stream: {e}", exc_info=True)
                    raise

            # If we're here, all retries failed
            raise OllamaStreamTimeoutError(
                f"Ollama streaming failed after {self.max_retries + 1} attempts: {last_error}")

        finally:
            # Clean up tracking
            self._active_streams.pop(request_id, None)

    async def _stream_attempt(self,
                              ollama_client,
                              model_name: str,
                              messages: list,
                              options: Optional[Dict[str, Any]],
                              request_id: str) -> AsyncGenerator[str, None]:
        """Single attempt at streaming with timeout protection."""

        start_time = time.time()
        last_chunk_time = start_time
        chunk_count = 0

        # Create the stream with total timeout
        try:
            stream_task = asyncio.create_task(
                self._create_ollama_stream(ollama_client, model_name, messages, options)
            )

            # Wait for stream creation with timeout
            stream_iterator = await asyncio.wait_for(stream_task, timeout=10.0)

        except asyncio.TimeoutError:
            raise OllamaStreamTimeoutError("Timeout creating Ollama stream")

        # Process stream chunks with individual timeouts
        try:
            while True:
                current_time = time.time()

                # Check total timeout
                if current_time - start_time > self.total_timeout:
                    raise OllamaStreamTimeoutError(f"Total streaming time exceeded {self.total_timeout}s")

                # Check chunk timeout
                if current_time - last_chunk_time > self.chunk_timeout:
                    raise OllamaStreamTimeoutError(f"No chunk received in {self.chunk_timeout}s")

                try:
                    # Get next chunk with timeout
                    chunk_task = asyncio.create_task(self._get_next_chunk(stream_iterator))
                    chunk_data = await asyncio.wait_for(chunk_task, timeout=self.chunk_timeout)

                    if chunk_data is None:  # Stream ended
                        logger.info(f"Ollama stream ended normally after {chunk_count} chunks")
                        break

                    # Process chunk
                    content = self._extract_content_from_chunk(chunk_data)
                    if content:
                        yield content
                        chunk_count += 1
                        last_chunk_time = current_time

                        # Update tracking
                        if request_id in self._active_streams:
                            self._active_streams[request_id].update({
                                'last_chunk_time': current_time,
                                'chunk_count': chunk_count,
                                'total_chars': self._active_streams[request_id]['total_chars'] + len(content)
                            })

                    # Check if stream is done
                    if hasattr(chunk_data, 'done') and chunk_data.done:
                        logger.info(f"Ollama stream marked as done after {chunk_count} chunks")
                        break

                except asyncio.TimeoutError:
                    raise OllamaStreamTimeoutError(f"Timeout waiting for next chunk after {chunk_count} chunks")

                # Brief pause to prevent tight loop
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
            logger.info(f"Ollama stream cancelled for {request_id}")
            raise
        except Exception as e:
            logger.error(f"Error in Ollama stream processing: {e}", exc_info=True)
            raise

    async def _create_ollama_stream(self,
                                    ollama_client,
                                    model_name: str,
                                    messages: list,
                                    options: Optional[Dict[str, Any]]):
        """Create the Ollama stream iterator."""
        try:
            return ollama_client.chat(
                model=model_name,
                messages=messages,
                stream=True,
                options=options or {}
            )
        except Exception as e:
            logger.error(f"Failed to create Ollama stream: {e}")
            raise

    async def _get_next_chunk(self, stream_iterator):
        """Get the next chunk from the stream iterator."""
        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, next, stream_iterator)
        except StopIteration:
            return None  # Stream ended normally
        except Exception as e:
            logger.error(f"Error getting next chunk: {e}")
            raise

    def _extract_content_from_chunk(self, chunk_data) -> Optional[str]:
        """Extract text content from an Ollama chunk."""
        try:
            # Handle different chunk formats
            if hasattr(chunk_data, 'message') and chunk_data.message:
                if hasattr(chunk_data.message, 'content'):
                    return chunk_data.message.content

            # Alternative format
            if isinstance(chunk_data, dict):
                message = chunk_data.get('message', {})
                if isinstance(message, dict):
                    return message.get('content', '')

            return None

        except Exception as e:
            logger.warning(f"Error extracting content from chunk: {e}")
            return None

    @asynccontextmanager
    async def stream_context(self, request_id: str):
        """Context manager for stream lifecycle management."""
        logger.debug(f"Starting stream context for {request_id}")
        try:
            yield
        except Exception as e:
            logger.error(f"Error in stream context {request_id}: {e}")
            raise
        finally:
            logger.debug(f"Ending stream context for {request_id}")
            self._active_streams.pop(request_id, None)

    def get_stream_stats(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get statistics for an active stream."""
        return self._active_streams.get(request_id)

    def get_all_stream_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all active streams."""
        return self._active_streams.copy()

    def cancel_stream(self, request_id: str) -> bool:
        """Cancel a specific stream."""
        if request_id in self._active_streams:
            self._active_streams.pop(request_id)
            logger.info(f"Cancelled stream {request_id}")
            return True
        return False

    def cancel_all_streams(self):
        """Cancel all active streams."""
        count = len(self._active_streams)
        self._active_streams.clear()
        logger.info(f"Cancelled {count} active streams")


# Integration helper for existing ollama_adapter.py
class OllamaStreamIntegration:
    """Helper to integrate the stream handler with existing OllamaAdapter."""

    @staticmethod
    def patch_ollama_adapter_streaming(ollama_adapter_instance):
        """Patch an existing OllamaAdapter instance to use the robust streaming."""

        # Store original method
        original_get_response_stream = ollama_adapter_instance.get_response_stream

        # Create stream handler
        stream_handler = OllamaStreamHandler(
            chunk_timeout=20.0,  # Shorter timeout for individual chunks
            total_timeout=180.0,  # 3 minutes total
            max_retries=2
        )

        async def enhanced_get_response_stream(history, options=None):
            """Enhanced streaming method with timeout handling."""

            # Prepare messages
            messages_for_api = ollama_adapter_instance._format_history_for_api(history)
            ollama_api_options = {}

            if options:
                if "temperature" in options and isinstance(options["temperature"], (float, int)):
                    ollama_api_options["temperature"] = float(options["temperature"])

            request_id = f"ollama_{int(time.time())}"

            try:
                logger.info(f"Starting enhanced Ollama stream for {ollama_adapter_instance._model_name}")

                async for chunk in stream_handler.stream_with_timeout(
                        ollama_client=ollama_adapter_instance._sync_client,
                        model_name=ollama_adapter_instance._model_name,
                        messages=messages_for_api,
                        options=ollama_api_options,
                        request_id=request_id
                ):
                    yield chunk

            except OllamaStreamTimeoutError as e:
                logger.error(f"Ollama streaming timeout: {e}")
                ollama_adapter_instance._last_error = f"Streaming timeout: {e}"
                yield f"[SYSTEM ERROR: Ollama streaming timed out - {e}]"

            except Exception as e:
                logger.error(f"Enhanced Ollama streaming error: {e}", exc_info=True)
                ollama_adapter_instance._last_error = f"Streaming error: {e}"
                yield f"[SYSTEM ERROR: {e}]"

        # Replace the method
        ollama_adapter_instance.get_response_stream = enhanced_get_response_stream
        ollama_adapter_instance._stream_handler = stream_handler

        logger.info("OllamaAdapter enhanced with robust streaming")
        return stream_handler