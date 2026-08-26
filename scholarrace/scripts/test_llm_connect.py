"""Quick LLM connectivity test — verifies that all 4 configured models
(Qwen, DeepSeek, GLM, StrongJudge) can actually respond to a simple prompt.

Usage:
    cd scholarrace && python -m scripts.test_llm_connect
"""

import asyncio
import sys
import time

# Ensure app package is importable
sys.path.insert(0, ".")

from app.agents.base import (
    create_qwen_provider,
    create_deepseek_provider,
    create_glm_provider,
    create_strong_judge_provider,
)
from app.config import get_settings


async def test_one(name: str, factory):
    """Test a single LLM provider with a trivial prompt."""
    print(f"\n--- {name} ---")
    settings = get_settings()
    if name == "Qwen":
        key = settings.qwen_api_key
        model = settings.qwen_model
        url = settings.qwen_base_url
    elif name == "DeepSeek":
        key = settings.deepseek_api_key
        model = settings.deepseek_model
        url = settings.deepseek_base_url
    elif name == "GLM":
        key = settings.glm_api_key
        model = settings.glm_model
        url = settings.glm_base_url
    else:  # StrongJudge
        key = settings.strong_model_api_key
        model = settings.strong_model_name
        url = settings.strong_model_base_url

    if not key:
        print(f"  SKIP: no API key configured")
        return False

    print(f"  model: {model}")
    print(f"  base_url: {url}")
    print(f"  key: {key[:12]}...{key[-4:]}")

    try:
        provider = factory()
        t0 = time.time()
        resp = await provider.generate(
            prompt="Say 'hello' in one word.",
            temperature=0.0,
        )
        elapsed = time.time() - t0
        if resp.success:
            print(f"  OK ({elapsed:.1f}s): {resp.content[:100]}")
            print(f"  tokens={resp.token_usage}")
            return True
        else:
            print(f"  FAIL ({elapsed:.1f}s): {resp.error}")
            return False
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


async def main():
    settings = get_settings()
    print(f"APP_ENV={settings.app_env}")
    print(f"is_test={settings.is_test}")

    results = {}
    results["Qwen"] = await test_one("Qwen", create_qwen_provider)
    results["DeepSeek"] = await test_one("DeepSeek", create_deepseek_provider)
    results["GLM"] = await test_one("GLM", create_glm_provider)
    results["StrongJudge"] = await test_one("StrongJudge", create_strong_judge_provider)

    print("\n========== Summary ==========")
    for name, ok in results.items():
        status = "OK" if ok else "FAIL"
        print(f"  {name:15s} {status}")

    all_ok = all(results.values())
    print(f"\n{'All passed!' if all_ok else 'Some failed — check above.'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
