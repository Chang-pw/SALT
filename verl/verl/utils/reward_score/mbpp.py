# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Reward function for MBPP dataset
"""

import json
import traceback
import signal
from contextlib import contextmanager


class TimeoutException(Exception):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager for timeout."""
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def extract_code(completion):
    """Extract Python code from completion string."""
    # Try to extract code from markdown code blocks
    if "```python" in completion:
        code = completion.split("```python")[-1].split("```")[0]
    elif "```" in completion:
        code = completion.split("```")[1].split("```")[0]
    else:
        # Assume the whole completion is code
        code = completion
    return code.strip()


def run_test(code, test_case, setup_code="", timeout=5):
    """Run a single test case and return whether it passed."""
    try:
        # Create a clean namespace for execution
        exec_globals = {}
        
        # Execute setup code if provided
        if setup_code:
            exec(setup_code, exec_globals)
        
        # Execute the solution code
        with time_limit(timeout):
            exec(code, exec_globals)
        
        # Execute the test case (assert statement)
        with time_limit(timeout):
            exec(test_case, exec_globals)
        
        return True
    except AssertionError:
        return False
    except TimeoutException:
        return False
    except Exception as e:
        return False


def compute_score(completion, test_cases_str, continuous=True):
    """
    Compute the score for MBPP by running test cases.
    
    Args:
        completion: The model's generated code
        test_cases_str: JSON string containing test_list and test_setup_code
        continuous: If True, return the fraction of passed tests; if False, return 1 only if all pass
    
    Returns:
        score: float between 0 and 1
        metadata: dict with test results
    """
    try:
        # Parse test cases
        if isinstance(test_cases_str, str):
            test_cases = json.loads(test_cases_str)
        else:
            test_cases = test_cases_str
        
        test_list = test_cases.get("test_list", [])
        setup_code = test_cases.get("test_setup_code", "")
        
        if not test_list:
            return 0.0, {"error": "No test cases provided"}
        
        # Extract code from completion
        code = extract_code(completion)
        
        # Run each test
        results = []
        for test_case in test_list:
            passed = run_test(code, test_case, setup_code)
            results.append(passed)
        
        # Calculate score
        num_passed = sum(results)
        num_total = len(results)
        
        if continuous:
            score = num_passed / num_total
        else:
            score = 1.0 if num_passed == num_total else 0.0
        
        metadata = {
            "num_passed": num_passed,
            "num_total": num_total,
            "results": results,
            "extracted_code": code[:500] if len(code) > 500 else code
        }
        
        return score, metadata
        
    except Exception as e:
        traceback.print_exc()
        return 0.0, {"error": str(e)}


if __name__ == "__main__":
    # Test the reward function
    test_completion = '''
```python
def first_repeated_char(str1):
    for index, c in enumerate(str1):
        if str1[:index+1].count(c) > 1:
            return c 
    return "None"
```
'''
    
    test_cases = {
        "test_list": [
            'assert first_repeated_char("abcabc") == "a"',
            'assert first_repeated_char("abc") == "None"',
            'assert first_repeated_char("123123") == "1"'
        ],
        "test_setup_code": ""
    }
    
    score, metadata = compute_score(test_completion, json.dumps(test_cases))
    print(f"Score: {score}")
    print(f"Metadata: {metadata}")
