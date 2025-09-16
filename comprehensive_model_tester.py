#!/usr/bin/env python3
"""
Comprehensive Model Testing Script
Tests all models from /v1/models endpoint one by one and generates working_models.json
"""

import asyncio
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from collections import defaultdict
import aiohttp
import os

class ComprehensiveModelTester:
    def __init__(self, api_base_url: str = "http://localhost:5000"):
        self.api_base_url = api_base_url
        self.session = None
        self.test_results = []
        self.working_models = []
        self.failed_models = []
        self.default_fallback_count = 0
        self.error_stats = defaultdict(int)
        self.provider_stats = defaultdict(lambda: {"success": 0, "failed": 0})
        
        # Setup logging
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_filename = f"model_test_log_{timestamp}.log"
        
        # Configure logging to both file and console
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_filename, encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def fetch_all_models(self) -> List[Dict[str, Any]]:
        """Fetch all models from /v1/models endpoint"""
        try:
            async with self.session.get(f"{self.api_base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    self.logger.info(f"✅ Fetched {len(models)} models from /v1/models endpoint")
                    return models
                else:
                    self.logger.error(f"❌ Failed to fetch models: HTTP {response.status}")
                    return []
        except Exception as e:
            self.logger.error(f"❌ Error fetching models: {e}")
            return []

    def determine_model_capability(self, model_id: str) -> str:
        """Determine model capability based on model ID patterns"""
        model_lower = model_id.lower()
        
        # Image generation models
        if any(keyword in model_lower for keyword in ['dalle', 'flux', 'midjourney', 'stable', 'sd', 'img', 'image']):
            return 'image'
        
        # Voice/Audio models
        if any(keyword in model_lower for keyword in ['voice', 'audio', 'tts', 'speech', 'whisper']):
            return 'voice'
        
        # Code models
        if any(keyword in model_lower for keyword in ['coder', 'code', 'programming']):
            return 'code'
        
        # Multimodal models
        if any(keyword in model_lower for keyword in ['multimodal', 'omni', 'vision', 'vl', 'vlm']):
            return 'multimodal'
        
        # Default to text for others
        return 'text'

    def generate_test_message(self, capability: str) -> Dict[str, Any]:
        """Generate appropriate test message based on model capability"""
        if capability == 'image':
            return {
                "messages": [{"role": "user", "content": "Generate a simple image of a cat"}],
                "max_tokens": 50
            }
        elif capability == 'voice':
            return {
                "messages": [{"role": "user", "content": "Convert this text to speech: Hello world"}],
                "max_tokens": 50
            }
        elif capability == 'code':
            return {
                "messages": [{"role": "user", "content": "Write a simple Python function to add two numbers"}],
                "max_tokens": 100
            }
        else:
            # Default text test
            return {
                "messages": [{"role": "user", "content": "Hello! Please respond with just 'Hi there!' to test this model."}],
                "max_tokens": 20
            }

    async def test_single_model(self, model_info: Dict[str, Any]) -> Dict[str, Any]:
        """Test a single model and return detailed results"""
        model_id = model_info['id']
        capability = self.determine_model_capability(model_id)
        test_request = self.generate_test_message(capability)
        test_request["model"] = model_id
        
        start_time = time.time()
        
        self.logger.info(f"🔄 Testing model: {model_id} (capability: {capability})")
        
        try:
            async with self.session.post(
                f"{self.api_base_url}/v1/chat/completions",
                json=test_request,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    try:
                        data = await response.json()
                        content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                        usage = data.get('usage', {})
                        
                        # Check if response fell back to default (PollinationsAI with openai model)
                        is_default_fallback = self.is_default_fallback(content, model_id)
                        if is_default_fallback:
                            self.default_fallback_count += 1
                        
                        result = {
                            "model_id": model_id,
                            "status": "success" if not is_default_fallback else "default_fallback",
                            "capability": capability,
                            "response_time": round(response_time, 2),
                            "tokens_used": usage.get('total_tokens', 0),
                            "prompt_tokens": usage.get('prompt_tokens', 0),
                            "completion_tokens": usage.get('completion_tokens', 0),
                            "response_preview": content[:100] + "..." if len(content) > 100 else content,
                            "response_length": len(content),
                            "is_working": not is_default_fallback,
                            "error": None,
                            "owned_by": model_info.get('owned_by', 'unknown'),
                            "created": model_info.get('created', int(time.time()))
                        }
                        
                        if not is_default_fallback:
                            self.logger.info(f"✅ SUCCESS: {model_id} - {response_time:.2f}s - {len(content)} chars")
                            self.working_models.append(result)
                            self.provider_stats[result['owned_by']]['success'] += 1
                        else:
                            self.logger.warning(f"⚠️  DEFAULT FALLBACK: {model_id} - fell back to PollinationsAI")
                            self.failed_models.append(result)
                            self.error_stats['default_fallback'] += 1
                        
                    except Exception as e:
                        result = {
                            "model_id": model_id,
                            "status": "parse_error",
                            "capability": capability,
                            "response_time": round(response_time, 2),
                            "tokens_used": 0,
                            "response_preview": "",
                            "response_length": 0,
                            "is_working": False,
                            "error": f"JSON parse error: {str(e)}",
                            "owned_by": model_info.get('owned_by', 'unknown'),
                            "created": model_info.get('created', int(time.time()))
                        }
                        
                        self.logger.error(f"❌ PARSE ERROR: {model_id} - {str(e)}")
                        self.failed_models.append(result)
                        self.error_stats['parse_error'] += 1
                        self.provider_stats[result['owned_by']]['failed'] += 1
                        
                else:
                    error_text = await response.text()
                    result = {
                        "model_id": model_id,
                        "status": f"http_{response.status}",
                        "capability": capability,
                        "response_time": round(response_time, 2),
                        "tokens_used": 0,
                        "response_preview": "",
                        "response_length": 0,
                        "is_working": False,
                        "error": error_text[:200],
                        "owned_by": model_info.get('owned_by', 'unknown'),
                        "created": model_info.get('created', int(time.time()))
                    }
                    
                    self.logger.error(f"❌ HTTP {response.status}: {model_id} - {error_text[:100]}")
                    self.failed_models.append(result)
                    self.error_stats[f'http_{response.status}'] += 1
                    self.provider_stats[result['owned_by']]['failed'] += 1
                    
        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            result = {
                "model_id": model_id,
                "status": "timeout",
                "capability": capability,
                "response_time": 30.0,
                "tokens_used": 0,
                "response_preview": "",
                "response_length": 0,
                "is_working": False,
                "error": "Request timed out after 30s",
                "owned_by": model_info.get('owned_by', 'unknown'),
                "created": model_info.get('created', int(time.time()))
            }
            
            self.logger.error(f"⏰ TIMEOUT: {model_id} - 30s timeout")
            self.failed_models.append(result)
            self.error_stats['timeout'] += 1
            self.provider_stats[result['owned_by']]['failed'] += 1
            
        except Exception as e:
            response_time = time.time() - start_time
            result = {
                "model_id": model_id,
                "status": "error",
                "capability": capability,
                "response_time": round(response_time, 2),
                "tokens_used": 0,
                "response_preview": "",
                "response_length": 0,
                "is_working": False,
                "error": str(e)[:200],
                "owned_by": model_info.get('owned_by', 'unknown'),
                "created": model_info.get('created', int(time.time()))
            }
            
            self.logger.error(f"❌ ERROR: {model_id} - {str(e)}")
            self.failed_models.append(result)
            self.error_stats['connection_error'] += 1
            self.provider_stats[result['owned_by']]['failed'] += 1
        
        self.test_results.append(result)
        return result

    def is_default_fallback(self, response_content: str, model_id: str) -> bool:
        """Check if response came from default fallback (PollinationsAI)"""
        # Common indicators that response came from PollinationsAI fallback
        fallback_indicators = [
            "I'm Claude" in response_content,
            "I'm an AI assistant" in response_content and len(response_content) > 100,
            # Add more specific patterns if needed
        ]
        
        # If the response is very generic and long, it might be from fallback
        if len(response_content) > 200 and any(indicator for indicator in fallback_indicators):
            return True
            
        return False

    async def run_comprehensive_test(self, delay_between_tests: float = 1.0):
        """Run comprehensive test on all models one by one"""
        self.logger.info("🚀 Starting Comprehensive Model Testing")
        self.logger.info("=" * 80)
        
        # Check API availability
        try:
            async with self.session.get(f"{self.api_base_url}/health") as response:
                if response.status != 200:
                    self.logger.error(f"❌ API not accessible (status: {response.status})")
                    return
        except Exception as e:
            self.logger.error(f"❌ API not accessible: {e}")
            return
        
        self.logger.info("✅ API server is running")
        
        # Fetch all models
        models_data = await self.fetch_all_models()
        if not models_data:
            self.logger.error("❌ No models found")
            return
        
        self.logger.info(f"\n🔄 Testing {len(models_data)} models one by one...")
        self.logger.info("=" * 80)
        
        # Test models one by one with delay
        for i, model_info in enumerate(models_data, 1):
            model_id = model_info['id']
            
            self.logger.info(f"\n[{i:3d}/{len(models_data)}] Testing: {model_id}")
            
            result = await self.test_single_model(model_info)
            
            # Add delay between tests to avoid overwhelming the API
            if delay_between_tests > 0 and i < len(models_data):
                await asyncio.sleep(delay_between_tests)
        
        # Generate final report
        await self.generate_final_report()
        
        # Create working_models.json
        await self.create_working_models_json()

    async def generate_final_report(self):
        """Generate comprehensive final report"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 COMPREHENSIVE TEST RESULTS")
        self.logger.info("=" * 80)
        
        total_models = len(self.test_results)
        working_count = len(self.working_models)
        failed_count = len(self.failed_models)
        success_rate = (working_count / total_models * 100) if total_models > 0 else 0
        
        self.logger.info(f"\n📈 OVERALL STATISTICS")
        self.logger.info(f"   Total Models Tested: {total_models}")
        self.logger.info(f"   ✅ Working Models: {working_count} ({success_rate:.1f}%)")
        self.logger.info(f"   ❌ Failed Models: {failed_count} ({100-success_rate:.1f}%)")
        self.logger.info(f"   ⚠️  Default Fallbacks: {self.default_fallback_count}")
        
        # Error breakdown
        if self.error_stats:
            self.logger.info(f"\n❌ ERROR BREAKDOWN")
            for error_type, count in sorted(self.error_stats.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_models * 100)
                self.logger.info(f"   {error_type}: {count} models ({percentage:.1f}%)")
        
        # Provider statistics
        self.logger.info(f"\n🏢 PROVIDER STATISTICS")
        self.logger.info(f"{'Provider':<20} {'Success':<8} {'Failed':<8} {'Total':<8} {'Rate':<8}")
        self.logger.info("-" * 60)
        
        for provider, stats in sorted(self.provider_stats.items(), key=lambda x: x[1]['success'], reverse=True):
            success = stats['success']
            failed = stats['failed']
            total = success + failed
            rate = (success / total * 100) if total > 0 else 0
            
            self.logger.info(f"{provider:<20} {success:<8} {failed:<8} {total:<8} {rate:>6.1f}%")
        
        # Capability breakdown
        capability_stats = defaultdict(lambda: {"working": 0, "failed": 0})
        for result in self.test_results:
            capability = result['capability']
            if result['is_working']:
                capability_stats[capability]['working'] += 1
            else:
                capability_stats[capability]['failed'] += 1
        
        self.logger.info(f"\n🎯 CAPABILITY BREAKDOWN")
        self.logger.info(f"{'Capability':<15} {'Working':<8} {'Failed':<8} {'Total':<8} {'Rate':<8}")
        self.logger.info("-" * 55)
        
        for capability, stats in sorted(capability_stats.items(), key=lambda x: x[1]['working'], reverse=True):
            working = stats['working']
            failed = stats['failed']
            total = working + failed
            rate = (working / total * 100) if total > 0 else 0
            
            self.logger.info(f"{capability:<15} {working:<8} {failed:<8} {total:<8} {rate:>6.1f}%")
        
        # Top working models
        if self.working_models:
            self.logger.info(f"\n🏆 TOP 10 FASTEST WORKING MODELS")
            sorted_working = sorted(self.working_models, key=lambda x: x['response_time'])[:10]
            for i, model in enumerate(sorted_working, 1):
                self.logger.info(f"   {i:2d}. {model['model_id']:<30} {model['response_time']:>6.2f}s ({model['owned_by']})")

    async def create_working_models_json(self):
        """Create working_models.json in the same format as models.json"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"working_models_{timestamp}.json"
        
        # Convert working models to models.json format
        working_models_data = {
            "object": "list",
            "data": []
        }
        
        for model in self.working_models:
            model_entry = {
                "id": model['model_id'],
                "object": "model",
                "created": model['created'],
                "owned_by": model['owned_by'],
                "capability": model['capability'],
                "performance": {
                    "response_time": model['response_time'],
                    "tokens_used": model['tokens_used'],
                    "response_length": model['response_length']
                },
                "test_info": {
                    "tested_at": datetime.now().isoformat(),
                    "status": model['status'],
                    "is_working": model['is_working']
                }
            }
            working_models_data['data'].append(model_entry)
        
        # Save working models JSON
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(working_models_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"\n💾 Working models saved to: {filename}")
            self.logger.info(f"   📊 {len(working_models_data['data'])} working models saved")
            
            # Also create a summary file
            summary_filename = f"test_summary_{timestamp}.json"
            summary_data = {
                "test_completed_at": datetime.now().isoformat(),
                "total_models_tested": len(self.test_results),
                "working_models": len(self.working_models),
                "failed_models": len(self.failed_models),
                "success_rate": (len(self.working_models) / len(self.test_results) * 100) if self.test_results else 0,
                "default_fallback_count": self.default_fallback_count,
                "error_statistics": dict(self.error_stats),
                "provider_statistics": dict(self.provider_stats),
                "log_file": self.log_filename,
                "working_models_file": filename
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"   📋 Test summary saved to: {summary_filename}")
            
        except Exception as e:
            self.logger.error(f"\n❌ Failed to save working models: {e}")

async def main():
    """Main function with configurable options"""
    import sys
    
    # Parse command line arguments
    delay_between_tests = 1.0  # Default 1 second delay
    
    if len(sys.argv) > 1:
        try:
            delay_between_tests = float(sys.argv[1])
        except ValueError:
            print("Usage: python comprehensive_model_tester.py [delay_seconds]")
            print("Example: python comprehensive_model_tester.py 0.5")
            return
    
    print(f"🚀 Starting comprehensive model testing with {delay_between_tests}s delay between tests")
    
    async with ComprehensiveModelTester() as tester:
        await tester.run_comprehensive_test(delay_between_tests)

if __name__ == "__main__":
    asyncio.run(main())
