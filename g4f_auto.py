#!/usr/bin/env python3
"""
G4F Auto API - Optimized OpenAI-compatible API with proper model ID preservation
Handles correct logging, streaming, provider rotation, and response optimization
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Union, AsyncGenerator, Any
from collections import defaultdict
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

# FastAPI app
app = FastAPI(
    title="G4F Auto API",
    description="Optimized OpenAI-compatible API with proper model ID preservation and provider rotation",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Free provider list (no authentication required)
FREE_PROVIDERS = [
    "PollinationsAI",
    "DeepInfra", 
    "WeWordle",
    "Yqcloud",
    "GLM",
    "Qwen",
    "CohereForAI_C4AI_Command",
    "LambdaChat",
    "OIVSCodeSer0501",
    "OIVSCodeSer2",
    "HuggingSpace",
    "AnyProvider"
]

# Provider priority order (most reliable first)
PROVIDER_PRIORITY = [
    "PollinationsAI",
    "DeepInfra",
    "WeWordle", 
    "Yqcloud",
    "AnyProvider",
    "GLM",
    "Qwen",
    "CohereForAI_C4AI_Command",
    "LambdaChat",
    "OIVSCodeSer2",
    "OIVSCodeSer0501",
    "HuggingSpace"
]

# Model aliases for common requests
MODEL_ALIASES = {
    "gpt-3.5-turbo": "gpt-4o-mini",
    "gpt-3.5": "gpt-4o-mini",
    "claude": "command-r",
    "claude-3": "command-r",
    "gemini": "gemini-2.5-flash-lite",
    "llama": "llama-3.3-70b",
    "deepseek": "deepseek-v3",
    "qwen": "qwen-max-latest"
}

def determine_model_capability(model_id: str) -> str:
    """Determine model capability based on model ID patterns"""
    model_lower = model_id.lower()
    
    if any(keyword in model_lower for keyword in ['dalle', 'flux', 'midjourney', 'stable', 'sd', 'img', 'image']):
        return 'image'
    elif any(keyword in model_lower for keyword in ['voice', 'audio', 'tts', 'speech', 'whisper']):
        return 'voice'
    elif any(keyword in model_lower for keyword in ['coder', 'code', 'programming']):
        return 'code'
    elif any(keyword in model_lower for keyword in ['multimodal', 'omni', 'vision', 'vl', 'vlm']):
        return 'multimodal'
    else:
        return 'text'

def discover_free_models_from_providers() -> Dict[str, List[str]]:
    """Dynamically discover which models each free provider actually supports"""
    provider_models = {}
    
    for provider_name in FREE_PROVIDERS:
        try:
            if provider_name == "AnyProvider":
                continue  # Skip AnyProvider as it's a fallback
                
            provider_class = getattr(g4f.Provider, provider_name, None)
            if provider_class and hasattr(provider_class, 'models'):
                models = getattr(provider_class, 'models', [])
                if models:
                    # Convert to list if it's not already
                    if hasattr(models, '__iter__') and not isinstance(models, str):
                        model_list = list(models)
                    else:
                        model_list = [str(models)]
                    
                    provider_models[provider_name] = model_list
                    logger.info(f"📋 {provider_name}: {len(model_list)} models")
                else:
                    logger.debug(f"⚠️ {provider_name}: No models attribute")
            else:
                logger.debug(f"❌ {provider_name}: Provider not found or no models")
                
        except Exception as e:
            logger.debug(f"❌ Error checking {provider_name}: {e}")
            
    return provider_models

def build_model_provider_map() -> Dict[str, List[str]]:
    """Build mapping of model_id -> list of providers that support it"""
    model_provider_map = defaultdict(list)
    provider_models = discover_free_models_from_providers()
    
    # Build reverse mapping: model -> providers
    for provider_name, models in provider_models.items():
        for model in models:
            model_id = str(model).lower().strip()
            if model_id:
                model_provider_map[model_id].append(provider_name)
    
    # Add AnyProvider as fallback for all models
    for model_id in model_provider_map.keys():
        if "AnyProvider" not in model_provider_map[model_id]:
            model_provider_map[model_id].append("AnyProvider")
    
    logger.info(f"🔗 Built model-provider mapping for {len(model_provider_map)} models")
    return dict(model_provider_map)

# Discover available models dynamically
logger.info("🔍 Discovering free models from providers...")
PROVIDER_MODELS = discover_free_models_from_providers()
MODEL_PROVIDER_MAP = build_model_provider_map()

# Load additional models from JSON if available
AVAILABLE_MODELS = {}
try:
    with open('models.json', 'r') as f:
        models_data = json.load(f)
        for model_info in models_data.get('data', []):
            model_id = model_info.get('id')
            if model_id:
                AVAILABLE_MODELS[model_id] = {
                    'id': model_id,
                    'object': 'model',
                    'created': model_info.get('created', int(time.time())),
                    'owned_by': model_info.get('owned_by', 'g4f'),
                    'capability': determine_model_capability(model_id)
                }
    logger.info(f"✅ Loaded {len(AVAILABLE_MODELS)} additional models from models.json")
except Exception as e:
    logger.warning(f"⚠️ Could not load models.json: {e}")

# Merge discovered models with JSON models
for model_id in MODEL_PROVIDER_MAP.keys():
    if model_id not in AVAILABLE_MODELS:
        AVAILABLE_MODELS[model_id] = {
            'id': model_id,
            'object': 'model',
            'created': int(time.time()),
            'owned_by': 'g4f',
            'capability': determine_model_capability(model_id)
        }

logger.info(f"🎯 Total available models: {len(AVAILABLE_MODELS)}")

def normalize_model_name(model: str) -> str:
    """Normalize model name and resolve aliases"""
    model = model.lower().strip()
    
    # Handle provider/model format
    if "/" in model:
        _, model_name = model.split("/", 1)
        model_name = model_name.lower()
        return MODEL_ALIASES.get(model_name, model_name)
    
    return MODEL_ALIASES.get(model, model)

def get_free_providers_for_model(model_id: str) -> List[tuple]:
    """Get list of free providers that actually support the specific model ID"""
    normalized_model = normalize_model_name(model_id).lower()
    original_model = model_id.lower()
    
    providers = []
    
    # Check if model exists in our discovered provider-model mapping
    available_providers = MODEL_PROVIDER_MAP.get(normalized_model, MODEL_PROVIDER_MAP.get(original_model, []))
    
    if available_providers:
        # Use providers that actually support this model
        for provider_name in available_providers:
            try:
                if provider_name == "AnyProvider":
                    providers.append((None, model_id))  # Keep original model ID
                    continue
                    
                provider_class = getattr(g4f.Provider, provider_name, None)
                if provider_class:
                    # Special handling for specific providers
                    if provider_name == "GLM" and model_id.lower().startswith("glm-"):
                        glm_model = model_id.replace("glm-", "GLM-").replace("4.5", "4.5")
                        providers.append((provider_class, glm_model))
                    else:
                        providers.append((provider_class, model_id))  # Keep original model ID
                        
            except AttributeError:
                logger.debug(f"Provider {provider_name} not available")
                continue
    else:
        # Fallback: try AnyProvider for unknown models
        logger.info(f"⚠️ Model '{model_id}' not found in provider mappings, using AnyProvider fallback")
        providers.append((None, model_id))
    
    logger.info(f"🔄 Model '{model_id}' has {len(providers)} supporting providers: {[p[0].__name__ if p[0] else 'AnyProvider' for p in providers]}")
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

class StreamingError(Exception):
    """Custom exception for streaming errors"""
    def __init__(self, message: str, error_type: str = "service_unavailable", code: int = 503):
        self.message = message
        self.error_type = error_type
        self.code = code
        super().__init__(message)

async def try_providers(model: str, messages: List[dict], providers: List[tuple], **kwargs) -> str:
    """Try providers in order until one succeeds, preserving exact model ID"""
    last_error = None
    
    for i, (provider, provider_model) in enumerate(providers):
        try:
            provider_name = provider.__name__ if provider else "AnyProvider"
            logger.info(f"🔄 [{i+1}/{len(providers)}] Trying {provider_name} with model: {provider_model}")
            
            response = await asyncio.to_thread(
                g4f.ChatCompletion.create,
                model=provider_model,
                messages=messages,
                provider=provider,
                stream=False,
                **kwargs
            )
            
            if response and isinstance(response, str) and response.strip():
                logger.info(f"✅ SUCCESS: {provider_name} responded with {len(response)} chars")
                return response
            else:
                logger.warning(f"⚠️ Empty response from {provider_name}")
                last_error = f"Empty response from {provider_name}"
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"❌ {provider_name} failed: {error_msg[:100]}")
            last_error = error_msg
            continue
    
    # All providers failed
    raise HTTPException(
        status_code=503,
        detail=f"All {len(providers)} providers failed for model '{model}'. Last error: {last_error}"
    )

async def try_providers_stream(model: str, messages: List[dict], providers: List[tuple], **kwargs) -> AsyncGenerator[str, None]:
    """Try providers for streaming, preserving exact model ID"""
    last_error = None
    
    for i, (provider, provider_model) in enumerate(providers):
        try:
            provider_name = provider.__name__ if provider else "AnyProvider"
            logger.info(f"🔄 [{i+1}/{len(providers)}] Streaming {provider_name} with model: {provider_model}")
            
            response = g4f.ChatCompletion.create(
                model=provider_model,
                messages=messages,
                provider=provider,
                stream=True,
                **kwargs
            )
            
            chunk_count = 0
            for chunk in response:
                if chunk and isinstance(chunk, str) and chunk.strip():
                    chunk_count += 1
                    if chunk_count == 1:
                        logger.info(f"✅ STREAMING SUCCESS: {provider_name}")
                    yield chunk
            
            if chunk_count > 0:
                return  # Success, exit function
            else:
                logger.warning(f"⚠️ No chunks from {provider_name}")
                last_error = f"No chunks from {provider_name}"
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"❌ Streaming {provider_name} failed: {error_msg[:100]}")
            last_error = error_msg
            continue
    
    # All providers failed
    raise StreamingError(
        f"All {len(providers)} providers failed for streaming model '{model}'. Last error: {last_error}",
        "service_unavailable",
        503
    )

@app.get("/")
async def root():
    return {
        "message": "G4F Auto API - Optimized with Model ID Preservation",
        "version": "2.0.0",
        "features": [
            "Exact model ID preservation",
            "Free provider rotation", 
            "Optimized error handling",
            "Streaming support",
            "OpenAI compatibility"
        ],
        "endpoints": {
            "models": "/v1/models",
            "chat_completions": "/v1/chat/completions",
            "health": "/health"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy", 
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(AVAILABLE_MODELS),
        "free_providers": len(FREE_PROVIDERS)
    }

@app.get("/v1/models")
async def list_models():
    """List all available models"""
    models_list = []
    for model_info in AVAILABLE_MODELS.values():
        models_list.append(ModelInfo(**model_info))
    
    return ModelsResponse(data=models_list)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion with exact model ID preservation"""
    try:
        # Get the exact model requested (preserve original case/format)
        requested_model = request.model
        
        # Get providers for this model
        providers = get_free_providers_for_model(requested_model)
        
        if not providers:
            raise HTTPException(
                status_code=400,
                detail=f"No providers available for model: {requested_model}"
            )
        
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Prepare kwargs
        kwargs = {}
        if request.temperature is not None:
            kwargs['temperature'] = request.temperature
        if request.max_tokens is not None:
            kwargs['max_tokens'] = request.max_tokens
        if request.top_p is not None:
            kwargs['top_p'] = request.top_p
        
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
                        "model": requested_model,  # Preserve exact model name
                        "choices": [{
                            "index": 0,
                            "delta": {"role": "assistant", "content": ""},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(initial_chunk)}\n\n"
                    
                    # Stream content
                    async for chunk in try_providers_stream(requested_model, messages, providers, **kwargs):
                        if chunk.strip():
                            chunk_data = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": requested_model,  # Preserve exact model name
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
                        "model": requested_model,  # Preserve exact model name
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }
                    yield f"data: {json.dumps(final_chunk)}\n\n"
                    yield "data: [DONE]\n\n"
                    
                except StreamingError as e:
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
            content = await try_providers(requested_model, messages, providers, **kwargs)
            
            # Create OpenAI-compatible response with exact model name
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=requested_model,  # Preserve exact model name
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
        logger.error(f"❌ Error in chat completion: {str(e)}")
        return JSONResponse(
            status_code=500,
            content=create_openai_error(
                "Internal server error",
                "internal_server_error",
                None,
                500
            )
        )

if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting G4F Auto API with model ID preservation...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
