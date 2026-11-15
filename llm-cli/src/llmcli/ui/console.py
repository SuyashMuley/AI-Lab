"""Output hygiene: data on stdout, usage and diagnostics on stderr."""

from rich.console import Console

from llmcli.core.models import Usage

out = Console()
err = Console(stderr=True)


def format_usage(usage: Usage, cost: float | None = None) -> str:
    parts = [f"tokens: {usage.input_tokens} in / {usage.output_tokens} out"]
    if usage.cached_tokens:
        parts[0] += f" ({usage.cached_tokens} cached)"
    if usage.latency_ms is not None:
        secs = usage.latency_ms / 1000
        parts.append(f"{secs:.2f}s")
        if usage.output_tokens and secs > 0:
            parts.append(f"{usage.output_tokens / secs:.1f} tok/s")
    if cost is not None:
        parts.append(f"~${cost:.4f} (estimate)")
    return " | ".join(parts)
