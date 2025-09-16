import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, AsyncGenerator, Any

import g4f
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request Models (OpenAI Compatible)
class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatCompletionRequest(BaseModel):
    model: str = Field(..., description="Model ID to use for the completion")
    messages: List[ChatMessage] = Field(..., description="List of messages for the conversation")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, description="Temperature for randomness")
    stream: Optional[bool] = Field(False, description="Whether to stream responses")
    top_p: Optional[float] = Field(1.0, description="Top-p sampling parameter")
    presence_penalty: Optional[float] = Field(0.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, description="Frequency penalty")


# Response Models (OpenAI Compatible)
class ChatChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatChoice]
    usage: ChatUsage


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# Load provider to model mapping from working models list
async def fetch_working_models():
    """Fetch working models from GitHub repository"""
    working_url = "https://raw.githubusercontent.com/maruf009sultan/g4f-working/refs/heads/main/working/working_results.txt"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(working_url)
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                return lines
    except Exception as e:
        logger.error(f"Failed to fetch working models: {e}")
    return []


def parse_working_models(lines: List[str]) -> Dict[str, List[tuple]]:
    """Parse working models and build PROVIDER_MODEL_MAP"""
    PROVIDER_MODEL_MAP = {}
    excluded_providers = {"HuggingSpace", "PerplexityLabs"}
    
    # Parse each line
    for line in lines:
        if not line.strip():
            continue
        
        parts = line.strip().split('|')
        if len(parts) >= 3:
            provider, model_id, modality = parts[0], parts[1], parts[2]
            
            # Skip excluded providers
            if provider in excluded_providers:
                continue
            
            # Add to PROVIDER_MODEL_MAP
            if model_id not in PROVIDER_MODEL_MAP:
                PROVIDER_MODEL_MAP[model_id] = []
            
            # Avoid duplicates
            if (provider, model_id) not in PROVIDER_MODEL_MAP[model_id]:
                PROVIDER_MODEL_MAP[model_id].append((provider, model_id))
    
    # Add default fallback
    PROVIDER_MODEL_MAP["default"] = [("AnyProvider", "default"), ("PollinationsAI", "openai")]
    
    return PROVIDER_MODEL_MAP


# Load working models at startup
try:
    import httpx
    import asyncio
    
    # For now, we'll load from the saved working models file
    # In production, you might want to fetch directly from GitHub
    try:
        with open('models.json', 'r') as f:
            models_data = json.load(f)
        
        # Build PROVIDER_MODEL_MAP from working models with proper provider grouping
        PROVIDER_MODEL_MAP = {}
        excluded_providers = {"HuggingSpace", "PerplexityLabs"}
        
        for model_info in models_data.get('data', []):
            model_id = model_info.get('id')
            main_provider = model_info.get('owned_by', 'AnyProvider')
            
            if model_id:
                if model_id not in PROVIDER_MODEL_MAP:
                    PROVIDER_MODEL_MAP[model_id] = []
                
                # Add main provider first (if not excluded)
                if main_provider not in excluded_providers:
                    # Special handling for GLM models - use uppercase model ID for GLM provider
                    if main_provider == "GLM" and model_id.startswith("glm-"):
                        glm_model_id = model_id.replace("glm-", "GLM-").replace("4.5", "4.5")
                        PROVIDER_MODEL_MAP[model_id].append((main_provider, glm_model_id))
                    else:
                        PROVIDER_MODEL_MAP[model_id].append((main_provider, model_id))
                
                # Add additional providers from the providers array
                additional_providers = model_info.get('providers', [])
                for additional in additional_providers:
                    additional_provider = additional.get('provider')
                    additional_model_id = additional.get('model_id', model_id)
                    
                    if additional_provider and additional_provider not in excluded_providers:
                        # Special handling for GLM models
                        if additional_provider == "GLM" and model_id.startswith("glm-"):
                            glm_model_id = model_id.replace("glm-", "GLM-").replace("4.5", "4.5")
                            if (additional_provider, glm_model_id) not in PROVIDER_MODEL_MAP[model_id]:
                                PROVIDER_MODEL_MAP[model_id].append((additional_provider, glm_model_id))
                        else:
                            if (additional_provider, additional_model_id) not in PROVIDER_MODEL_MAP[model_id]:
                                PROVIDER_MODEL_MAP[model_id].append((additional_provider, additional_model_id))
                
                # Always add AnyProvider as fallback if not already present
                any_provider_exists = any(provider == "AnyProvider" for provider, _ in PROVIDER_MODEL_MAP[model_id])
                if not any_provider_exists:
                    PROVIDER_MODEL_MAP[model_id].append(("AnyProvider", model_id))
        
        # Add default fallback
        PROVIDER_MODEL_MAP["default"] = [("AnyProvider", "default"), ("PollinationsAI", "openai")]
        
        logger.info(f"Loaded {len(PROVIDER_MODEL_MAP)} models from working models list")
        
        # Debug: Log first few models to verify loading
        sample_models = list(PROVIDER_MODEL_MAP.keys())[:10]
        logger.info(f"Sample models loaded: {sample_models}")
        
        # Debug: Check if common models exist
        test_models = ["gpt-4", "gpt-4o-mini", "deepseek-v3", "qwen-max-latest"]
        for test_model in test_models:
            if test_model in PROVIDER_MODEL_MAP:
                providers = [p[0] for p in PROVIDER_MODEL_MAP[test_model]]
                logger.info(f"Model '{test_model}' has providers: {providers}")
            else:
                logger.warning(f"Model '{test_model}' NOT FOUND in PROVIDER_MODEL_MAP")
        
    except FileNotFoundError:
        logger.warning("models.json not found, using default mapping")
        PROVIDER_MODEL_MAP = {
            "default": [("AnyProvider", "default"), ("PollinationsAI", "openai")]
        }
except Exception as e:
    logger.error(f"Error loading working models: {e}")
    PROVIDER_MODEL_MAP = {
        "default": [("AnyProvider", "default"), ("PollinationsAI", "openai")]
    }


# Model aliases for common naming variations
MODEL_ALIASES = {
    "gpt-4-turbo": "gpt-4",
    "gpt-4-turbo-preview": "gpt-4",
    "gpt-3.5-turbo": "gpt-4o-mini",
    "claude-3": "command-r",
    "claude-3-sonnet": "command-r",
    "claude-3-opus": "command-r-plus",
    "claude-3-haiku": "command-a",
    "gemini": "gemini-2.5-flash-lite",
    "gemini-pro": "gemini-2.5-flash-lite",
    "gemini-1.5-pro": "gemini-2.5-flash-lite",
    "llama-2": "llama-3.3-70b",
    "llama-3": "llama-3.3-70b",
    "llama-2-70b": "llama-3.3-70b",
    "llama-3-70b": "llama-3.3-70b",
    "mixtral": "mistral-small-3.1-24b",
    "mixtral-8x7b": "mistral-small-3.1-24b",
    "deepseek": "deepseek-v3",
    "qwen": "qwen-max-latest",
}


app = FastAPI(
    title="G4F OpenAI Compatible API",
    description="OpenAI-compatible API server using G4F with automatic provider routing",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def normalize_model_name(model: str) -> str:
    """Normalize model name and resolve aliases"""
    model = model.lower().strip()
    
    # Handle provider/model format (e.g., "deepseek/DeepSeek-R1-0528-Turbo")
    if "/" in model:
        provider_hint, model_name = model.split("/", 1)
        # Try to find the model without provider prefix first
        model_name = model_name.lower()
        return MODEL_ALIASES.get(model_name, model_name)
    
    return MODEL_ALIASES.get(model, model)


def get_available_providers(model: str) -> List[tuple]:
    """Get list of available (provider, model_id) pairs for a given model"""
    original_model = model.lower().strip()
    provider_hint = None
    
    # Extract provider hint if present (e.g., "deepseek/DeepSeek-R1-0528-Turbo")
    if "/" in original_model:
        provider_hint, _ = original_model.split("/", 1)
    
    normalized_model = normalize_model_name(model)
    provider_model_pairs = PROVIDER_MODEL_MAP.get(normalized_model, PROVIDER_MODEL_MAP["default"])
    
    # If we have a provider hint, prioritize that provider
    if provider_hint:
        prioritized_pairs = []
        other_pairs = []
        
        for provider_name, model_id in provider_model_pairs:
            if provider_name.lower() == provider_hint:
                prioritized_pairs.append((provider_name, model_id))
            else:
                other_pairs.append((provider_name, model_id))
        
        # Use prioritized pairs first, then fallback to others
        provider_model_pairs = prioritized_pairs + other_pairs
    
    # Convert provider names to actual provider objects and keep model IDs
    providers = []
    for provider_name, model_id in provider_model_pairs:
        try:
            # Handle special case for AnyProvider (use None to let g4f auto-select)
            if provider_name == "AnyProvider":
                providers.append((None, model_id))
                continue
                
            provider = getattr(g4f.Provider, provider_name, None)
            if provider:
                providers.append((provider, model_id))
            else:
                logger.warning(f"Provider {provider_name} not found in g4f.Provider, skipping")
                
        except AttributeError:
            logger.warning(f"Provider {provider_name} not found in g4f.Provider, skipping")
            continue
    
    # Ensure we have at least one provider
    if not providers:
        providers = [(None, model)]  # Let g4f auto-select
    
    return providers


def create_openai_error(message: str, error_type: str = "service_unavailable", param: str = None, code: int = None) -> dict:
    """Create OpenAI-compatible error response"""
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": param,
            "code": code
        }
    }

async def try_providers(model: str, messages: List[dict], providers: List[tuple], **kwargs) -> str:
    """Try providers in order until one succeeds"""
    for provider, provider_model in providers:
        try:
            logger.info(f"Trying provider: {provider} with model: {provider_model}")
            
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model=provider_model,
                messages=messages,
                provider=provider,
                stream=False
            )
            
            if response and isinstance(response, str):
                logger.info(f"Success with provider: {provider}")
                return response
                
        except Exception as e:
            logger.warning(f"Provider {provider} failed: {str(e)}")
            continue
    
    # Raise HTTPException, will be caught and converted to JSONResponse in the endpoint
    raise HTTPException(
        status_code=503,
        detail="All providers failed. Please try again later."
    )


class StreamingError(Exception):
    """Exception to signal streaming provider failure"""
    def __init__(self, message: str, error_type: str = "service_unavailable", code: int = 503):
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(message)

async def try_providers_stream(model: str, messages: List[dict], providers: List[tuple], **kwargs) -> AsyncGenerator[str, None]:
    """Try providers in order for streaming until one succeeds"""
    for provider, provider_model in providers:
        try:
            logger.info(f"Trying streaming provider: {provider} with model: {provider_model}")
            
            response = g4f.ChatCompletion.create(
                model=provider_model,
                messages=messages,
                provider=provider,
                stream=True
            )
            
            for chunk in response:
                if chunk and isinstance(chunk, str):
                    yield chunk
            return
                
        except Exception as e:
            logger.warning(f"Streaming provider {provider} failed: {str(e)}")
            continue
    
    # All providers failed - raise exception to be handled by generate()
    raise StreamingError(
        "All providers failed. Please try again later.",
        "service_unavailable",
        503
    )


@app.get("/")
async def root():
    return {
        "message": "G4F OpenAI Compatible API Server",
        "version": "1.0.0",
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions"
        }
    }


@app.get("/v1/models", response_model=ModelsResponse)
async def list_models():
    """List all available models"""
    models = []
    
    # Get all unique models from our mapping
    all_models = set(PROVIDER_MODEL_MAP.keys())
    all_models.update(MODEL_ALIASES.keys())
    all_models.discard("default")
    
    for model_id in sorted(all_models):
        models.append(ModelInfo(
            id=model_id,
            created=int(time.time()),
            owned_by="g4f"
        ))
    
    return ModelsResponse(data=models)


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create a chat completion (OpenAI compatible)"""
    try:
        # Normalize model name
        model = normalize_model_name(request.model)
        
        # Get available providers for this model
        providers = get_available_providers(model)
        
        if not providers:
            raise HTTPException(
                status_code=400,
                detail=f"No providers available for model: {request.model}"
            )
        
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        if request.stream:
            # Handle streaming response
            async def generate():
                request_id = f"chatcmpl-{int(time.time())}"
                
                try:
                    # Send initial chunk
                    initial_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    
                    # Stream content
                    full_content = ""
                    async for chunk in try_providers_stream(model, messages, providers):
                        if chunk.strip():
                            full_content += chunk
                            chunk_data = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": request.model,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"content": chunk},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {json.dumps(chunk_data)}\n\n"
                    
                    # Send final chunk
                    final_chunk = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": request.model,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except StreamingError as e:
                    # Handle streaming error - emit error event and terminate
                    error_data = {
                        "error": {
                            "message": e.message,
                            "type": e.error_type,
                            "code": e.code
                        }
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate(), 
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache", 
                    "Connection": "keep-alive",
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Headers": "*"
                }
            )
        
        else:
            # Handle non-streaming response
            content = await try_providers(model, messages, providers)
            
            # Create OpenAI-compatible response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=request.model,
                choices=[ChatChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=content),
                    finish_reason="stop"
                )],
                usage=ChatUsage(
                    prompt_tokens=sum(len(msg.content.split()) for msg in request.messages),
                    completion_tokens=len(content.split()),
                    total_tokens=sum(len(msg.content.split()) for msg in request.messages) + len(content.split())
                )
            )
            
            return response
            
    except HTTPException as he:
        # Return JSONResponse for OpenAI-compatible error format
        return JSONResponse(
            status_code=he.status_code,
            content=create_openai_error(
                str(he.detail) if hasattr(he, 'detail') else "Request failed",
                "invalid_request_error",
                None,
                he.status_code
            )
        )
    except Exception as e:
        logger.error(f"Error in chat completion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_openai_error(
                "Internal server error",
                "internal_server_error",
                None,
                500
            )
        )


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)