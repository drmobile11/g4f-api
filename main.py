#!/usr/bin/env python3
"""
G4F Optimized API - Using only curated working models from models_group.json
Implements proper provider rotation for each model with no authentication required
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
    category: Optional[str] = None

class ModelsResponse(BaseModel):
    object: str = "list"
    data: List[ModelInfo]

# FastAPI app
app = FastAPI(
    title="G4F Optimized API",
    description="Curated working models with free provider rotation - No authentication required",
    version="3.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Free providers that don't require authentication (ordered by reliability)
FREE_PROVIDERS = [
    "PollinationsAI",
    "WeWordle", 
    "Yqcloud",
    "CohereForAI_C4AI_Command",
    "LambdaChat",
    "OIVSCodeSer0501",
    "OIVSCodeSer2",
    "GLM",
    "Qwen",
    "DeepInfra",
    "AnyProvider"  # Fallback
]

# Load curated models from models_group.json
CURATED_MODELS = {}
MODEL_CATEGORIES = {}
AVAILABLE_MODELS = {}

def load_curated_models():
    """Load only curated working models from models_group.json"""
    global CURATED_MODELS, MODEL_CATEGORIES, AVAILABLE_MODELS
    
    try:
        with open('models_group.json', 'r') as f:
            groups = json.load(f)
        
        # Flatten all models and track categories
        all_models = set()
        for category, models in groups.items():
            for model in models:
                model_id = model.strip()
                all_models.add(model_id)
                MODEL_CATEGORIES[model_id] = category.replace('_models', '')
        
        # Create provider mapping for each curated model
        for model_id in all_models:
            CURATED_MODELS[model_id] = get_providers_for_model(model_id)
            AVAILABLE_MODELS[model_id] = {
                'id': model_id,
                'object': 'model',
                'created': int(time.time()),
                'owned_by': 'g4f',
                'category': MODEL_CATEGORIES.get(model_id, 'general')
            }
        
        logger.info(f"✅ Loaded {len(CURATED_MODELS)} curated models:")
        logger.info(f"   📝 Coding: {len(groups.get('coding_models', []))} models")
        logger.info(f"   💬 Chat: {len(groups.get('chat_models', []))} models") 
        logger.info(f"   🧠 Deep Thinking: {len(groups.get('deep_thinking_models', []))} models")
        
    except Exception as e:
        logger.error(f"❌ Failed to load models_group.json: {e}")
        # Fallback to basic models
        CURATED_MODELS = {
            "gpt-4": get_providers_for_model("gpt-4"),
            "command-r": get_providers_for_model("command-r")
        }
        AVAILABLE_MODELS = {
            "gpt-4": {'id': 'gpt-4', 'object': 'model', 'created': int(time.time()), 'owned_by': 'g4f', 'category': 'chat'},
            "command-r": {'id': 'command-r', 'object': 'model', 'created': int(time.time()), 'owned_by': 'g4f', 'category': 'chat'}
        }

def get_providers_for_model(model_id: str) -> List[tuple]:
    """Get free providers that support the specific model"""
    providers = []
    
    # Special handling for different model types
    if "gpt-4" in model_id.lower():
        # GPT-4 variants work with these providers
        provider_names = ["WeWordle", "Yqcloud", "PollinationsAI", "AnyProvider"]
    elif "command" in model_id.lower():
        # Command models work with Cohere providers
        provider_names = ["CohereForAI_C4AI_Command", "AnyProvider"]
    elif "deepseek" in model_id.lower():
        # DeepSeek models
        provider_names = ["DeepInfra", "AnyProvider"]
    elif "qwen" in model_id.lower():
        # Qwen models - based on working model analysis
        if model_id in ["Qwen/Qwen3-Coder-480B-A35B-Instruct", "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"]:
            provider_names = ["DeepInfra", "AnyProvider"]  # These work with DeepInfra
        elif model_id in ["qwen-3-14b", "qwen-3-30b", "qwen-3-32b"]:
            provider_names = ["AnyProvider"]  # These work with AnyProvider
        elif model_id == "qwq-32b":
            provider_names = ["Qwen", "AnyProvider"]  # This works with Qwen provider
        elif model_id == "qwen3-32b-fp8":
            provider_names = ["AnyProvider"]  # This works with g4f auto-selection
        else:
            provider_names = ["AnyProvider"]  # Fallback for other Qwen models
    elif "glm" in model_id.lower() or "chatglm" in model_id.lower():
        # GLM models
        provider_names = ["GLM", "AnyProvider"]
    elif "llama" in model_id.lower():
        # Llama models
        provider_names = ["DeepInfra", "LambdaChat", "AnyProvider"]
    elif "phi" in model_id.lower():
        # Phi models
        provider_names = ["DeepInfra", "AnyProvider"]
    else:
        # Default providers for other models
        provider_names = ["PollinationsAI", "DeepInfra", "AnyProvider"]
    
    # Convert provider names to actual provider objects
    for provider_name in provider_names:
        try:
            if provider_name == "AnyProvider":
                providers.append((None, model_id))  # None = let g4f auto-select
                continue
                
            provider_class = getattr(g4f.Provider, provider_name, None)
            if provider_class:
                # Special model ID handling for specific providers
                if provider_name == "GLM" and model_id.lower().startswith(("glm", "chatglm")):
                    # GLM provider expects uppercase format
                    if model_id.lower() == "chatglm":
                        glm_model = "ChatGLM"
                    elif "glm-4.5" in model_id.lower():
                        glm_model = "GLM-4.5"
                    else:
                        glm_model = model_id.replace("glm-", "GLM-")
                    providers.append((provider_class, glm_model))
                elif provider_name == "DeepInfra" and "/" not in model_id:
                    # DeepInfra might need full model path for some models
                    providers.append((provider_class, model_id))
                else:
                    providers.append((provider_class, model_id))
            else:
                logger.debug(f"Provider {provider_name} not found")
                
        except AttributeError:
            logger.debug(f"Provider {provider_name} not available")
            continue
    
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
    """Try providers in order until one succeeds"""
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
    """Try providers for streaming"""
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
                return  # Success
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

# Load curated models on startup
load_curated_models()

@app.get("/")
async def root():
    return {
        "message": "G4F Optimized API - Curated Working Models Only",
        "version": "3.0.0",
        "features": [
            "Curated working models only",
            "Free provider rotation",
            "No authentication required",
            "Model categorization",
            "OpenAI compatibility"
        ],
        "model_categories": {
            "coding": len([m for m, c in MODEL_CATEGORIES.items() if c == 'coding']),
            "chat": len([m for m, c in MODEL_CATEGORIES.items() if c == 'chat']),
            "deep_thinking": len([m for m, c in MODEL_CATEGORIES.items() if c == 'deep_thinking'])
        },
        "total_models": len(CURATED_MODELS),
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
        "curated_models": len(CURATED_MODELS),
        "free_providers": len(FREE_PROVIDERS)
    }

@app.get("/v1/models")
async def list_models():
    """List all curated models"""
    models_list = []
    for model_info in AVAILABLE_MODELS.values():
        models_list.append(ModelInfo(**model_info))
    
    return ModelsResponse(data=models_list)

@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion with curated models only"""
    try:
        requested_model = request.model
        
        # Check if model is in our curated list
        if requested_model not in CURATED_MODELS:
            raise HTTPException(
                status_code=400,
                detail=f"Model '{requested_model}' not available. Use /v1/models to see available models."
            )
        
        # Get providers for this curated model
        providers = CURATED_MODELS[requested_model]
        
        if not providers:
            raise HTTPException(
                status_code=500,
                detail=f"No providers configured for model: {requested_model}"
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
        
        category = MODEL_CATEGORIES.get(requested_model, 'general')
        logger.info(f"🎯 Using {category} model: {requested_model} with {len(providers)} providers")
        
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
                        "model": requested_model,
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
                                "model": requested_model,
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
                        "model": requested_model,
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
            
            # Create OpenAI-compatible response
            response = ChatCompletionResponse(
                id=f"chatcmpl-{int(time.time())}",
                created=int(time.time()),
                model=requested_model,
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
    logger.info("🚀 Starting G4F Optimized API with curated models...")
    uvicorn.run(app, host="0.0.0.0", port=5000)
