#!/usr/bin/env python3
"""
Model Identity Tester - Tests all models from /v1/models endpoint
Asks each model "what is your model" to verify identity and functionality
Uses non-streaming requests to avoid chunking conflicts
"""

import asyncio
import aiohttp
import json
import logging
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import argparse

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_identity_test_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelIdentityTester:
    def __init__(self, base_url: str = "http://localhost:5000", wait_between_tests: float = 2.0):
        self.base_url = base_url.rstrip('/')
        self.wait_between_tests = wait_between_tests
        self.session = None
        
        # Test statistics
        self.stats = {
            'total_models': 0,
            'working_models': 0,
            'failed_models': 0,
            'start_time': None,
            'end_time': None,
            'total_duration': 0
        }
        
        # Results storage
        self.working_models = []
        self.failed_models = []
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60),
            connector=aiohttp.TCPConnector(limit=10)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def fetch_available_models(self) -> List[Dict[str, Any]]:
        """Fetch all available models from /v1/models endpoint"""
        try:
            logger.info(f"[FETCH] Fetching models from {self.base_url}/v1/models")
            
            async with self.session.get(f"{self.base_url}/v1/models") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get('data', [])
                    logger.info(f"[SUCCESS] Found {len(models)} models")
                    return models
                else:
                    error_text = await response.text()
                    logger.error(f"[ERROR] Failed to fetch models: HTTP {response.status} - {error_text}")
                    return []
                    
        except Exception as e:
            logger.error(f"[ERROR] Error fetching models: {e}")
            return []
    
    async def test_model_identity(self, model_info: Dict[str, Any], index: int) -> Dict[str, Any]:
        """Test a single model by asking 'what is your model'"""
        model_id = model_info.get('id', 'unknown')
        
        test_result = {
            'index': index,
            'model_id': model_id,
            'model_info': model_info,
            'test_timestamp': datetime.now().isoformat(),
            'status': 'unknown',
            'response_time': 0.0,
            'response_content': '',
            'response_length': 0,
            'tokens_used': 0,
            'error': None,
            'is_working': False,
            'prompt_tokens': 0,
            'completion_tokens': 0
        }
        
        # Test payload - non-streaming to avoid chunking conflicts
        payload = {
            "model": model_id,
            "messages": [
                {
                    "role": "user", 
                    "content": "What is your model? Please respond with just your model name or identifier."
                }
            ],
            "max_tokens": 100,
            "temperature": 0.1,
            "stream": False  # Explicitly disable streaming
        }
        
        logger.info(f"[TEST] [{index}/{self.stats['total_models']}] Testing model: {model_id}")
        
        start_time = time.time()
        
        try:
            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                
                response_time = time.time() - start_time
                test_result['response_time'] = round(response_time, 2)
                
                if response.status == 200:
                    response_data = await response.json()
                    
                    # Extract response content
                    choices = response_data.get('choices', [])
                    if choices and len(choices) > 0:
                        message = choices[0].get('message', {})
                        content = message.get('content', '').strip()
                        test_result['response_content'] = content
                        test_result['response_length'] = len(content)
                        
                        # Extract token usage
                        usage = response_data.get('usage', {})
                        test_result['prompt_tokens'] = usage.get('prompt_tokens', 0)
                        test_result['completion_tokens'] = usage.get('completion_tokens', 0)
                        test_result['tokens_used'] = usage.get('total_tokens', 0)
                        
                        # Check if response is meaningful
                        if content and len(content) > 0:
                            test_result['status'] = 'success'
                            test_result['is_working'] = True
                            logger.info(f"[SUCCESS] {model_id} - SUCCESS: '{content[:100]}{'...' if len(content) > 100 else ''}'")
                        else:
                            test_result['status'] = 'empty_response'
                            test_result['error'] = "Empty or no response content"
                            logger.warning(f"[WARNING] {model_id} - Empty response")
                    else:
                        test_result['status'] = 'no_choices'
                        test_result['error'] = "No choices in response"
                        logger.warning(f"[WARNING] {model_id} - No choices in response")
                        
                else:
                    # Handle HTTP errors
                    error_text = await response.text()
                    test_result['status'] = f'http_{response.status}'
                    test_result['error'] = error_text
                    logger.warning(f"[ERROR] {model_id} - HTTP {response.status}: {error_text[:200]}")
                    
        except asyncio.TimeoutError:
            test_result['response_time'] = time.time() - start_time
            test_result['status'] = 'timeout'
            test_result['error'] = 'Request timeout'
            logger.warning(f"[TIMEOUT] {model_id} - Timeout after {test_result['response_time']:.1f}s")
            
        except Exception as e:
            test_result['response_time'] = time.time() - start_time
            test_result['status'] = 'error'
            test_result['error'] = str(e)
            logger.error(f"[ERROR] {model_id} - Error: {e}")
        
        return test_result
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run identity tests on all available models"""
        logger.info("[START] Starting Model Identity Testing")
        self.stats['start_time'] = datetime.now().isoformat()
        
        # Fetch all models
        models = await self.fetch_available_models()
        if not models:
            logger.error("[ERROR] No models found to test")
            return self.generate_final_report()
        
        self.stats['total_models'] = len(models)
        logger.info(f"[INFO] Testing {len(models)} models with {self.wait_between_tests}s intervals")
        
        # Test each model
        for index, model_info in enumerate(models, 1):
            test_result = await self.test_model_identity(model_info, index)
            
            # Categorize results
            if test_result['is_working']:
                self.working_models.append(test_result)
                self.stats['working_models'] += 1
            else:
                self.failed_models.append(test_result)
                self.stats['failed_models'] += 1
            
            # Wait between tests to avoid overwhelming the API
            if index < len(models):
                logger.info(f"[WAIT] Waiting {self.wait_between_tests}s before next test...")
                await asyncio.sleep(self.wait_between_tests)
        
        self.stats['end_time'] = datetime.now().isoformat()
        return self.generate_final_report()
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if self.stats['start_time'] and self.stats['end_time']:
            start = datetime.fromisoformat(self.stats['start_time'])
            end = datetime.fromisoformat(self.stats['end_time'])
            self.stats['total_duration'] = (end - start).total_seconds()
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_models_tested': self.stats['total_models'],
                'working_models': self.stats['working_models'],
                'failed_models': self.stats['failed_models'],
                'success_rate': round((self.stats['working_models'] / max(self.stats['total_models'], 1)) * 100, 2),
                'total_duration_seconds': round(self.stats['total_duration'], 2),
                'average_time_per_model': round(self.stats['total_duration'] / max(self.stats['total_models'], 1), 2)
            },
            'working_models': {
                'count': len(self.working_models),
                'models': self.working_models
            },
            'failed_models': {
                'count': len(self.failed_models),
                'models': self.failed_models
            },
            'statistics': {
                'status_breakdown': self._get_status_breakdown(),
                'response_time_stats': self._get_response_time_stats(),
                'token_usage_stats': self._get_token_usage_stats()
            }
        }
        
        return report
    
    def _get_status_breakdown(self) -> Dict[str, int]:
        """Get breakdown of test statuses"""
        breakdown = {}
        all_results = self.working_models + self.failed_models
        
        for result in all_results:
            status = result.get('status', 'unknown')
            breakdown[status] = breakdown.get(status, 0) + 1
            
        return breakdown
    
    def _get_response_time_stats(self) -> Dict[str, float]:
        """Get response time statistics"""
        all_results = self.working_models + self.failed_models
        response_times = [r.get('response_time', 0) for r in all_results if r.get('response_time', 0) > 0]
        
        if not response_times:
            return {'min': 0, 'max': 0, 'average': 0}
            
        return {
            'min': round(min(response_times), 2),
            'max': round(max(response_times), 2),
            'average': round(sum(response_times) / len(response_times), 2)
        }
    
    def _get_token_usage_stats(self) -> Dict[str, int]:
        """Get token usage statistics"""
        working_results = [r for r in self.working_models if r.get('tokens_used', 0) > 0]
        
        if not working_results:
            return {'total_tokens': 0, 'average_tokens': 0}
            
        total_tokens = sum(r.get('tokens_used', 0) for r in working_results)
        
        return {
            'total_tokens': total_tokens,
            'average_tokens': round(total_tokens / len(working_results), 2)
        }
    
    def save_results(self, report: Dict[str, Any]):
        """Save test results to JSON files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save complete report
        report_file = f"model_identity_report_{timestamp}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info(f"[SAVE] Complete report saved: {report_file}")
        
        # Save working models only
        working_file = f"working_models_identity_{timestamp}.json"
        working_data = {
            'timestamp': report['test_summary']['timestamp'],
            'total_working': len(self.working_models),
            'models': self.working_models
        }
        with open(working_file, 'w', encoding='utf-8') as f:
            json.dump(working_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[SAVE] Working models saved: {working_file}")
        
        # Save failed models only
        failed_file = f"failed_models_identity_{timestamp}.json"
        failed_data = {
            'timestamp': report['test_summary']['timestamp'],
            'total_failed': len(self.failed_models),
            'models': self.failed_models
        }
        with open(failed_file, 'w', encoding='utf-8') as f:
            json.dump(failed_data, f, indent=2, ensure_ascii=False)
        logger.info(f"[SAVE] Failed models saved: {failed_file}")

async def main():
    """Main function to run the model identity tester"""
    parser = argparse.ArgumentParser(description='Test model identity for all available models')
    parser.add_argument('--url', default='http://localhost:5000', help='Base URL of the API server')
    parser.add_argument('--wait', type=float, default=2.0, help='Wait time between tests in seconds')
    
    args = parser.parse_args()
    
    logger.info("[INIT] Model Identity Tester Starting")
    logger.info(f"[CONFIG] API URL: {args.url}")
    logger.info(f"[CONFIG] Wait between tests: {args.wait}s")
    
    async with ModelIdentityTester(args.url, args.wait) as tester:
        try:
            # Run all tests
            report = await tester.run_all_tests()
            
            # Save results
            tester.save_results(report)
            
            # Print summary
            summary = report['test_summary']
            logger.info("=" * 60)
            logger.info("[COMPLETE] MODEL IDENTITY TEST COMPLETE")
            logger.info("=" * 60)
            logger.info(f"[STATS] Total Models Tested: {summary['total_models_tested']}")
            logger.info(f"[STATS] Working Models: {summary['working_models']}")
            logger.info(f"[STATS] Failed Models: {summary['failed_models']}")
            logger.info(f"[STATS] Success Rate: {summary['success_rate']}%")
            logger.info(f"[STATS] Total Duration: {summary['total_duration_seconds']}s")
            logger.info(f"[STATS] Avg Time/Model: {summary['average_time_per_model']}s")
            logger.info("=" * 60)
            
            # Print status breakdown
            if report['statistics']['status_breakdown']:
                logger.info("[BREAKDOWN] Status Breakdown:")
                for status, count in report['statistics']['status_breakdown'].items():
                    logger.info(f"   {status}: {count}")
            
            return 0
            
        except KeyboardInterrupt:
            logger.info("[INTERRUPT] Test interrupted by user")
            return 1
        except Exception as e:
            logger.error(f"[ERROR] Unexpected error: {e}")
            return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))
