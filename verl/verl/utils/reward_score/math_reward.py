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

try:
    from math_verify.errors import TimeoutException
    from math_verify.metric import math_metric
    from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
except ImportError:
    print("To use Math-Verify, please install it first by running `pip install math-verify`.")

def sanitize_model_output(s: str) -> str:
    # 把 Python 把 \b 解析成的退格符，修回成 LaTeX 需要的 \b...
    return s.replace("\x08", r"\b")

def _extract_numeric_tail(s: str):
    import re

    nums = re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))
    return nums[-1] if nums else None


def compute_score(model_output: str, ground_truth: str, timeout_score: float = 0) -> dict:
    """Math-Verify scorer with a numeric fallback.

    Returns a dict for compatibility with upstream logging:
        {"score": 1/-1, "acc": bool, "pred": extracted_pred}
    """
    model_output = sanitize_model_output(model_output)
    verify_func = math_metric(
        gold_extraction_target=(LatexExtractionConfig(),),
        pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
    )
    ret_score = 0.0
    pred_extracted = None

    # Wrap the ground truth in \boxed{} format for verification
    ground_truth_boxed = "\\boxed{" + ground_truth + "}"
    try:
        ret_score, preds = verify_func([ground_truth_boxed], [model_output])
        if preds:
            pred_extracted = preds[0]
    except TimeoutException:
        ret_score = timeout_score
    except Exception:
        pass

    # Fallback: simple numeric tail match (handles "#### 18" / "\boxed{18}" / plain numbers)
    if ret_score <= 0:
        gt_num = _extract_numeric_tail(ground_truth)
        pred_num = _extract_numeric_tail(model_output)
        if gt_num is not None and pred_num is not None and gt_num == pred_num:
            ret_score = 1.0
            pred_extracted = pred_num

    # Normalize pred_extracted to a simple string/number (avoid list -> numpy mean error)
    if isinstance(pred_extracted, (list, tuple)):
        # pick the last element if it's a list of candidate strings/numbers
        pred_extracted = pred_extracted[-1] if len(pred_extracted) > 0 else None

    # Ensure incorrect answers return 0
    if ret_score <= 0:
        ret_score = 0.0

    acc = ret_score > 0
    return {"score": float(ret_score), "acc": acc, "pred": pred_extracted}

if __name__ == "__main__":
    model_output = '''\boxed{18}'''
    ground_truth = "18"
    print(compute_score(model_output, ground_truth))