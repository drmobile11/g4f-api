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


# Provider to Model Mapping with provider-specific model IDs
PROVIDER_MODEL_MAP = {
    # OpenAI-like Models
    "gpt-4": [("WeWordle", "gpt-4"), ("Yqcloud", "gpt-4"), ("AnyProvider", "gpt-4")],
    "gpt-4.1": [("OIVSCodeSer0501", "gpt-4.1-mini"), ("AnyProvider", "gpt-4.1")],
    "gpt-4o-mini": [("OIVSCodeSer2", "gpt-4o-mini"), ("AnyProvider", "gpt-4o-mini")],
    "gpt-4o-mini-tts": [("AnyProvider", "gpt-4o-mini-tts")],
    
    # DeepSeek Models
    "deepseek-v3": [("DeepInfra", "deepseek-ai/DeepSeek-V3"), ("AnyProvider", "deepseek-v3")],
    "deepseek-v3.1": [("DeepInfra", "deepseek-ai/DeepSeek-V3.1")],
    "deepseek-r1": [("DeepInfra", "deepseek-ai/DeepSeek-R1-0528"), ("AnyProvider", "deepseek-r1")],
    "deepseek-r1-0528": [("DeepInfra", "deepseek-ai/DeepSeek-R1-0528"), ("AnyProvider", "deepseek-r1-0528")],
    "deepseek-r1-0528-turbo": [("DeepInfra", "deepseek-ai/DeepSeek-R1-0528-Turbo"), ("AnyProvider", "deepseek-r1-0528-turbo")],
    "deepseek-r1-distill-llama-70b": [("DeepInfra", "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"), ("AnyProvider", "deepseek-r1-distill-llama-70b")],
    "deepseek-v3-0324": [("DeepInfra", "deepseek-ai/DeepSeek-V3-0324"), ("AnyProvider", "deepseek-v3-0324")],
    "deepseek-prover-v2": [("DeepInfra", "deepseek-ai/DeepSeek-Prover-V2-671B"), ("AnyProvider", "deepseek-prover-v2")],
    "deepseek-prover-v2-671b": [("DeepInfra", "deepseek-ai/DeepSeek-Prover-V2-671B"), ("AnyProvider", "deepseek-prover-v2-671b")],
    
    # Qwen Models
    "qwen-max-latest": [("Qwen", "qwen-max-latest")],
    "qwen3-max-preview": [("Qwen", "qwen3-max-preview")],
    "qwen3-coder-plus": [("Qwen", "qwen3-coder-plus")],
    "qwq-32b": [("Qwen", "qwq-32b")],
    "qwen-turbo-2025-02-11": [("Qwen", "qwen-turbo-2025-02-11")],
    "qwen2.5-omni-7b": [("Qwen", "qwen2.5-omni-7b")],
    "qwen2.5-14b-instruct-1m": [("Qwen", "qwen2.5-14b-instruct-1m")],
    "qwen2.5-72b-instruct": [("Qwen", "qwen2.5-72b-instruct")],
    "qwen-2.5-72b": [("AnyProvider", "qwen-2.5-72b")],
    "qwen-3-14b": [("AnyProvider", "qwen-3-14b")],
    "qwen-3-32b": [("AnyProvider", "qwen-3-32b")],
    "qwen-3-30b": [("AnyProvider", "qwen-3-30b")],
    "qwen-2.5-coder-32b": [("PollinationsAI", "qwen-2.5-coder-32b"), ("AnyProvider", "qwen-2.5-coder-32b")],
    "qwen-3-235b-a22b-2507": [("DeepInfra", "Qwen/Qwen3-235B-A22B-Instruct-2507"), ("AnyProvider", "qwen-3-235b-a22b-2507")],
    "qwen-3-coder-480b-a35b-turbo": [("DeepInfra", "Qwen/Qwen3-Coder-480B-A35B-Instruct-Turbo"), ("AnyProvider", "qwen-3-coder-480b-a35b-turbo")],
    
    # Llama Models
    "llama-3.3-70b": [("AnyProvider", "llama-3.3-70b")],
    "llama-3.3-70b-instruct": [("DeepInfra", "meta-llama/Llama-3.3-70B-Instruct")],
    "llama-3.3-70b-instruct-turbo": [("DeepInfra", "meta-llama/Llama-3.3-70B-Instruct-Turbo")],
    "llama-3.3-70b-instruct-fp8": [("LambdaChat", "llama3.3-70b-instruct-fp8")],
    "llama-3.2-90b": [("AnyProvider", "llama-3.2-90b")],
    "llama-4-scout": [("AnyProvider", "llama-4-scout")],
    "llama-4-scout-17b-16e-instruct": [("DeepInfra", "meta-llama/Llama-4-Scout-17B-16E-Instruct")],
    "llama-4-maverick-17b-128e-instruct": [("DeepInfra", "meta-llama/Llama-4-Maverick-17B-128E-Instruct")],
    "llama-4-maverick-17b-128e-instruct-turbo": [("DeepInfra", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-Turbo")],
    "llama-4-maverick-17b-128e-instruct-fp8": [("DeepInfra", "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8")],
    "llama-guard-4-12b": [("DeepInfra", "meta-llama/Llama-Guard-4-12B"), ("AnyProvider", "llama-guard-4-12b")],
    "hermes-3-llama-3.1-405b-fp8": [("LambdaChat", "hermes-3-llama-3.1-405b-fp8")],
    "hermes-3-llama-3.1-405b": [("AnyProvider", "hermes-3-llama-3.1-405b")],
    
    # GLM Models
    "glm-4-32b": [("GLM", "GLM-4-32B"), ("AnyProvider", "glm-4-32b")],
    "glm-4.5": [("GLM", "GLM-4.5"), ("AnyProvider", "glm-4.5")],
    "glm-4.5-air": [("GLM", "GLM-4.5-Air")],
    "chatglm": [("GLM", "ChatGLM"), ("AnyProvider", "chatglm")],
    "z1-rumination": [("GLM", "Z1-Rumination"), ("AnyProvider", "z1-rumination")],
    
    # Gemini Models
    "gemini-2.5-flash-lite": [("PollinationsAI", "gemini-2.5-flash-lite"), ("AnyProvider", "gemini-2.5-flash-lite")],
    
    # Gemma Models
    "gemma-3-27b-it": [("DeepInfra", "google/gemma-3-27b-it"), ("AnyProvider", "gemma-3-27b-it")],
    "gemma-3-12b-it": [("DeepInfra", "google/gemma-3-12b-it"), ("AnyProvider", "gemma-3-12b-it")],
    "gemma-3-4b-it": [("DeepInfra", "google/gemma-3-4b-it"), ("AnyProvider", "gemma-3-4b-it")],
    "gemma-1.1-7b": [("AnyProvider", "gemma-1.1-7b")],
    "gemma-2-9b": [("AnyProvider", "gemma-2-9b")],
    "gemma-2-27b": [("AnyProvider", "gemma-2-27b")],
    
    # Command Models
    "command-r": [("CohereForAI_C4AI_Command", "command-r"), ("HuggingSpace", "command-r"), ("AnyProvider", "command-r")],
    "command-r-plus": [("CohereForAI_C4AI_Command", "command-r-plus"), ("HuggingSpace", "command-r-plus"), ("AnyProvider", "command-r-plus")],
    "command-r-08-2024": [("CohereForAI_C4AI_Command", "command-r-08-2024"), ("HuggingSpace", "command-r-08-2024")],
    "command-r-plus-08-2024": [("CohereForAI_C4AI_Command", "command-r-plus-08-2024"), ("HuggingSpace", "command-r-plus-08-2024")],
    "command-r7b-12-2024": [("CohereForAI_C4AI_Command", "command-r7b-12-2024"), ("HuggingSpace", "command-r7b-12-2024")],
    "command-a-03-2025": [("CohereForAI_C4AI_Command", "command-a-03-2025"), ("HuggingSpace", "command-a-03-2025")],
    "command-r7b-arabic-02-2025": [("CohereForAI_C4AI_Command", "command-r7b-arabic-02-2025"), ("HuggingSpace", "command-r7b-arabic-02-2025")],
    "command-a": [("HuggingSpace", "command-a"), ("AnyProvider", "command-a")],
    "command-r7b": [("AnyProvider", "command-r7b")],
    
    # Mistral Models
    "mistral-small-3.1-24b": [("PollinationsAI", "mistral-small-3.1-24b"), ("AnyProvider", "mistral-small-3.1-24b")],
    "mistral-small-3.2-24b-instruct-2506": [("DeepInfra", "mistralai/Mistral-Small-3.2-24B-Instruct-2506")],
    "devstral-small-2507": [("DeepInfra", "mistralai/Devstral-Small-2507"), ("AnyProvider", "devstral-small-2507")],
    
    # Phi Models
    "phi-4": [("AnyProvider", "phi-4")],
    "phi-4-reasoning-plus": [("DeepInfra", "microsoft/phi-4-reasoning-plus"), ("AnyProvider", "phi-4-reasoning-plus")],
    "phi-4-multimodal-instruct": [("DeepInfra", "microsoft/Phi-4-multimodal-instruct")],
    "phi-4-multimodal": [("AnyProvider", "phi-4-multimodal")],
    
    # Meta AI
    "meta-ai": [("MetaAI", "meta-ai"), ("AnyProvider", "meta-ai")],
    
    # Perplexity Models
    "sonar": [("PerplexityLabs", "sonar"), ("AnyProvider", "sonar")],
    "sonar-pro": [("PerplexityLabs", "sonar-pro"), ("AnyProvider", "sonar-pro")],
    "sonar-reasoning": [("PerplexityLabs", "sonar-reasoning"), ("AnyProvider", "sonar-reasoning")],
    "sonar-reasoning-pro": [("PerplexityLabs", "sonar-reasoning-pro"), ("AnyProvider", "sonar-reasoning-pro")],
    
    # Opera Aria
    "aria": [("OperaAria", "aria"), ("AnyProvider", "aria")],
    
    # Special Models
    "mintlify": [("Mintlify", "mintlify")],
    "apriel-5b-instruct": [("LambdaChat", "apriel-5b-instruct")],
    "apriel-5b": [("AnyProvider", "apriel-5b")],
    
    # OpenAI Family Models
    "ash": [("OpenAIFM", "ash")],
    "shimmer": [("OpenAIFM", "shimmer")],
    "nova": [("OpenAIFM", "nova")],
    "coral": [("OpenAIFM", "coral")],
    "sage": [("OpenAIFM", "sage")],
    "friendly": [("OpenAIFM", "friendly")],
    "alloy": [("OpenAIFM", "alloy")],
    "echo": [("OpenAIFM", "echo")],
    "onyx": [("OpenAIFM", "onyx")],
    "fable": [("OpenAIFM", "fable")],
    
    # Default fallback (removed DeepInfra gpt-4 to avoid predictable failures)
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
    return MODEL_ALIASES.get(model, model)


def get_available_providers(model: str) -> List[tuple]:
    """Get list of available (provider, model_id) pairs for a given model"""
    normalized_model = normalize_model_name(model)
    provider_model_pairs = PROVIDER_MODEL_MAP.get(normalized_model, PROVIDER_MODEL_MAP["default"])
    
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
    uvicorn.run(app, host="0.0.0.0", port=8000)