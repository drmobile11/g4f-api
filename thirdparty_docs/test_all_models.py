#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import logging
from datetime import datetime
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:5000"
TEST_MESSAGE = "Hello, world! This is a test message. Please respond with a short greeting."
TIMEOUT_SECONDS = 30

async def get_available_models(session: aiohttp.ClientSession) -> List[str]:
    """Fetch the list of available models from the API"""
    try:
        async with session.get(f"{API_BASE_URL}/v1/models") as response:
            if response.status == 200:
                data = await response.json()
                models = [model['id'] for model in data['data']]
                logger.info(f"Found {len(models)} models")
                return models
            else:
                logger.error(f"Failed to fetch models. Status: {response.status}")
                return []
    except Exception as e:
        logger.error(f"Error fetching models: {str(e)}")
        return []

async def test_model(session: aiohttp.ClientSession, model: str) -> Dict[str, Any]:
    """Test a single model and return results"""
    start_time = time.time()
    result = {
        "model": model,
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "latency": None,
        "timeout": False,
        "error": None,
        "response": None
    }
    
    try:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": TEST_MESSAGE}],
            "stream": False,
            "temperature": 0.7,
            "max_tokens": 100
        }
        
        async with session.post(f"{API_BASE_URL}/v1/chat/completions", 
                                json=payload, 
                                timeout=aiohttp.ClientTimeout(total=TIMEOUT_SECONDS)) as response:
            end_time = time.time()
            result["latency"] = end_time - start_time
            
            if response.status == 200:
                data = await response.json()
                result["success"] = True
                result["response"] = data
                logger.info(f"✓ Model {model} succeeded in {result['latency']:.2f}s")
            else:
                error_data = await response.json()
                result["error"] = f"HTTP {response.status}: {error_data.get('error', 'Unknown error')}"
                logger.warning(f"✗ Model {model} failed with status {response.status}")
                
    except asyncio.TimeoutError:
        end_time = time.time()
        result["latency"] = end_time - start_time
        result["timeout"] = True
        result["error"] = f"Request timed out after {TIMEOUT_SECONDS} seconds"
        logger.warning(f"⏰ Model {model} timed out after {TIMEOUT_SECONDS}s")
        
    except Exception as e:
        end_time = time.time()
        result["latency"] = end_time - start_time
        result["error"] = str(e)
        logger.error(f"✗ Model {model} failed with exception: {str(e)}")
    
    return result

async def test_all_models():
    """Test all models and save results to logs.json"""
    results = []
    
    async with aiohttp.ClientSession() as session:
        # Check if server is healthy
        try:
            async with session.get(f"{API_BASE_URL}/health") as response:
                if response.status == 200:
                    logger.info("Server is healthy")
                else:
                    logger.error("Server is not healthy. Aborting tests.")
                    return
        except Exception as e:
            logger.error(f"Cannot connect to server: {str(e)}. Aborting tests.")
            return
        
        # Get list of models
        models = await get_available_models(session)
        if not models:
            logger.error("No models found. Aborting tests.")
            return
        
        logger.info(f"Starting tests for {len(models)} models")
        
        # Test each model
        for i, model in enumerate(models, 1):
            logger.info(f"Testing model {i}/{len(models)}: {model}")
            result = await test_model(session, model)
            results.append(result)
            
            # Save results after each test to avoid losing data
            with open("logs.json", "w") as f:
                json.dump(results, f, indent=2)
        
        logger.info(f"Completed testing {len(models)} models. Results saved to logs.json")
        
        # Summary
        successful = sum(1 for r in results if r["success"])
        timeouts = sum(1 for r in results if r["timeout"])
        failures = len(results) - successful - timeouts
        
        logger.info(f"Summary: {successful} successful, {timeouts} timeouts, {failures} failures")
        
        # Detailed summary
        if failures > 0:
            failed_models = [r["model"] for r in results if not r["success"] and not r["timeout"]]
            logger.info(f"Failed models: {', '.join(failed_models)}")
        
        if timeouts > 0:
            timed_out_models = [r["model"] for r in results if r["timeout"]]
            logger.info(f"Timed out models: {', '.join(timed_out_models)}")

if __name__ == "__main__":
    asyncio.run(test_all_models())
