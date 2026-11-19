"""
LLM Client implementations for LangPSRO

Supports various LLM providers through OpenAI-compatible API.

Includes per-call and per-tag token usage tracking. Each call to the
underlying API returns a `usage` object; we accumulate them on the client
so a run can later report a per-component breakdown:

    client = DeepSeekChatClient()
    client(prompt, tag="rollout")
    client(prompt, tag="brs_analysis")
    client(prompt, tag="brs_improve")
    print(client.get_usage_summary())
"""

import os
from pathlib import Path
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from openai import OpenAI
from dotenv import load_dotenv

_MODEL_DEFAULTS = {
    "deepseek": ("DEEPSEEK_MODEL", "deepseek-chat"),
    "volcengine": ("VOLCENGINE_MODEL", "deepseek-v4-flash"),
}


def _load_run_environment() -> None:
    load_dotenv(Path.cwd() / ".env", override=False)


def resolve_model(provider: str, model: Optional[str] = None) -> str:
    """Resolve the model ID before logging or opening a provider client."""
    if provider not in _MODEL_DEFAULTS:
        raise ValueError(f"Unsupported provider: {provider!r}")
    key, default = _MODEL_DEFAULTS[provider]
    _load_run_environment()
    return model or os.environ.get(key) or default


def create_deepseek_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client configured for the DeepSeek-compatible API.

    Set DEEPSEEK_BASE_URL to override the endpoint. The endpoint must
    accept the selected model and its provider-specific reasoning setting.

    Args:
        api_key: API key. If None, reads from DEEPSEEK_API_KEY env var.

    Returns:
        OpenAI client configured for DeepSeek
    """
    _load_run_environment()
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DeepSeek API key not provided. "
            "Set DEEPSEEK_API_KEY environment variable or pass api_key argument."
        )

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


def create_volcengine_client(api_key: Optional[str] = None) -> OpenAI:
    """
    Create an OpenAI client configured for the Volcengine ARK API,
    which hosts a DeepSeek-family model (`deepseek-v4-flash`) on the
    OpenAI-compatible `/api/plan/v3` endpoint.

    Args:
        api_key: Volcengine API key. If None, reads from
            VOLCENGINE_API_KEY env var.

    Returns:
        OpenAI client pointed at the Volcengine ARK endpoint
    """
    _load_run_environment()
    api_key = api_key or os.environ.get("VOLCENGINE_API_KEY")
    if not api_key:
        raise ValueError(
            "Volcengine API key not provided. "
            "Set VOLCENGINE_API_KEY environment variable or pass api_key argument."
        )

    base_url = os.environ.get(
        "VOLCENGINE_BASE_URL", "https://ark.cn-beijing.volces.com/api/plan/v3"
    )

    return OpenAI(
        api_key=api_key,
        base_url=base_url
    )


@dataclass
class TokenUsage:
    """Aggregated token usage across one or more LLM calls."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0  # DeepSeek "thinking" tokens, when reported
    call_count: int = 0

    def add(self, other: "TokenUsage") -> None:
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.total_tokens += other.total_tokens
        self.reasoning_tokens += other.reasoning_tokens
        self.call_count += other.call_count

    def to_dict(self) -> Dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "call_count": self.call_count,
        }

    @classmethod
    def from_api(cls, usage: Any) -> "TokenUsage":
        """
        Build a TokenUsage from an OpenAI-style usage object.

        DeepSeek (and most OpenAI-compatible APIs) report
        prompt_tokens / completion_tokens / total_tokens, and may add a
        `completion_tokens_details.reasoning_tokens` field for thinking
        models. Both are read defensively.
        """
        if usage is None:
            return cls(call_count=1)
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or 0
        reasoning = 0
        details = getattr(usage, "completion_tokens_details", None)
        if details is not None:
            reasoning = getattr(details, "reasoning_tokens", 0) or 0
        return cls(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=total,
            reasoning_tokens=reasoning,
            call_count=1,
        )

    @staticmethod
    def zero() -> "TokenUsage":
        return TokenUsage()


class DeepSeekChatClient:
    """
    Wrapper for DeepSeek Chat API.

    Provides a simple interface for LLM calls needed by LangPSRO.

    Default max_tokens: 100K tokens.

    Tracks token usage per call. Pass `tag="..."` to categorize calls
    (e.g. "rollout", "brs_analysis", "brs_improve", "eval"); call
    `get_usage_summary()` to see a per-tag breakdown plus the grand total.
    """

    DEFAULT_MAX_TOKENS = 100_000  # 100K tokens

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        track_usage: bool = True,
        thinking_enabled: bool = False,
    ):
        """
        Initialize DeepSeek chat client.

        Args:
            model: Model name. If None, reads from DEEPSEEK_MODEL env var.
            api_key: API key (reads from env if None)
            temperature: Default sampling temperature (default: 0.2 — low for
                stable rollout/evaluation; can be overridden per-call via kwargs)
            max_tokens: Maximum tokens to generate (default: 100K)
            track_usage: Whether to record per-call token usage on the client.
                Disable for lightweight production runs.
            thinking_enabled: Whether to enable DeepSeek's built-in reasoning
                (extra_body={"thinking": {"type": "enabled"}}) for every
                call made on this client. Default False (reasoning disabled)
                to preserve current behavior. Per-call `thinking=...` kwarg
                on __call__/chat can shadow this default.
        """
        self.model = resolve_model("deepseek", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.track_usage = track_usage
        self.thinking_enabled = thinking_enabled

        # usage bookkeeping
        self._total_usage: TokenUsage = TokenUsage.zero()
        self._tagged_usage: Dict[str, TokenUsage] = {}
        self._call_log: List[Dict[str, Any]] = []

        self.client = create_deepseek_client(api_key)

    # ------------------------------------------------------------------
    # Usage API
    # ------------------------------------------------------------------
    def reset_usage(self) -> None:
        """Clear all accumulated usage statistics."""
        self._total_usage = TokenUsage.zero()
        self._tagged_usage = {}
        self._call_log = []

    def get_usage(self, tag: Optional[str] = None) -> TokenUsage:
        """
        Return aggregated token usage.

        Args:
            tag: If provided, return usage for that tag only. If None,
                return the grand total.
        """
        if tag is None:
            return self._total_usage
        return self._tagged_usage.get(tag, TokenUsage.zero())

    def get_usage_summary(self) -> Dict[str, Any]:
        """
        Return a nested dict with the grand total and per-tag breakdown:

            {
              "total": {prompt_tokens, completion_tokens, total_tokens, ...},
              "by_tag": {
                  "rollout": {...},
                  "brs_analysis": {...},
                  ...
              }
            }
        """
        return {
            "total": self._total_usage.to_dict(),
            "by_tag": {tag: usage.to_dict() for tag, usage in self._tagged_usage.items()},
        }

    def get_call_log(self) -> List[Dict[str, Any]]:
        """
        Return per-call records. Each entry contains:
            {tag, prompt_tokens, completion_tokens, total_tokens,
             reasoning_tokens, timestamp}
        Only populated when track_usage=True.
        """
        return list(self._call_log)

    def _record_usage(
        self,
        usage: TokenUsage,
        tag: Optional[str],
        thinking_enabled: bool = False,
    ) -> None:
        if not self.track_usage:
            return
        self._total_usage.add(usage)
        if tag is not None:
            self._tagged_usage.setdefault(tag, TokenUsage.zero()).add(usage)
        self._call_log.append({
            "tag": tag,
            "thinking_enabled": thinking_enabled,
            **usage.to_dict(),
        })

    # ------------------------------------------------------------------
    # Call API
    # ------------------------------------------------------------------
    def __call__(
        self,
        prompt: str,
        tag: Optional[str] = None,
        *,
        thinking: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Call the LLM with a prompt.

        Args:
            prompt: Input prompt
            tag: Optional category label for usage tracking
                (e.g. "rollout", "brs_analysis").
            thinking: Per-call override for reasoning mode. None falls
                back to the instance's `thinking_enabled`. True sends
                `extra_body={"thinking": {"type": "enabled"}}`.
            **kwargs: Per-call overrides (e.g. temperature=0.5 for strategy
                generation; falls back to instance default)

        Returns:
            LLM response text
        """
        use_thinking = self.thinking_enabled if thinking is None else thinking
        extra_body = {"thinking": {"type": "enabled" if use_thinking else "disabled"}}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body=extra_body,
        )
        self._record_usage(
            TokenUsage.from_api(getattr(response, "usage", None)),
            tag,
            thinking_enabled=use_thinking,
        )
        return response.choices[0].message.content

    def chat(
        self,
        messages: list,
        tag: Optional[str] = None,
        *,
        thinking: Optional[bool] = None,
        **kwargs,
    ) -> str:
        """
        Call the LLM with a chat-style message list.

        Args:
            messages: List of {"role": ..., "content": ...} messages
            tag: Optional category label for usage tracking.
            thinking: Per-call override for reasoning mode (see __call__).
            **kwargs: Additional API parameters

        Returns:
            LLM response text
        """
        use_thinking = self.thinking_enabled if thinking is None else thinking
        extra_body = {"thinking": {"type": "enabled" if use_thinking else "disabled"}}
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
            extra_body=extra_body,
        )
        self._record_usage(
            TokenUsage.from_api(getattr(response, "usage", None)),
            tag,
            thinking_enabled=use_thinking,
        )
        return response.choices[0].message.content


class VolcengineChatClient(DeepSeekChatClient):
    """
    Wrapper for the Volcengine ARK API hosting a DeepSeek-family
    model (`deepseek-v4-flash` by default).

    The ARK API is OpenAI-compatible at `/api/plan/v3` and the
    underlying model supports the same `extra_body={"thinking": ...}`
    reasoning toggle as the native DeepSeek endpoint. So this class
    inherits everything from `DeepSeekChatClient` (token tracking,
    per-call `thinking=` override, `get_usage_summary()`, etc.) and
    only overrides:

      * `__init__`: use `create_volcengine_client` and read
        `VOLCENGINE_API_KEY` / `VOLCENGINE_MODEL` / `VOLCENGINE_BASE_URL`
        env vars instead of the DeepSeek ones.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = DeepSeekChatClient.DEFAULT_MAX_TOKENS,
        track_usage: bool = True,
        thinking_enabled: bool = False,
    ):
        # Initialize state fields WITHOUT calling DeepSeekChatClient's
        # __init__ (which would build a DeepSeek client). Set up state
        # directly, mirroring the parent class.
        self.model = resolve_model("volcengine", model)
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.track_usage = track_usage
        self.thinking_enabled = thinking_enabled

        # Usage bookkeeping (same as parent).
        self._total_usage: TokenUsage = TokenUsage.zero()
        self._tagged_usage: Dict[str, TokenUsage] = {}
        self._call_log: List[Dict[str, Any]] = []

        self.client = create_volcengine_client(api_key)
