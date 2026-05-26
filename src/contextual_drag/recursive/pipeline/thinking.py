"""Parse a model response into (post_thinking_text, thinking_status).

Three formats recognized:

1. gpt-oss harmony WITH special tokens kept (vLLM's
   `skip_special_tokens=False` — what our config.py uses to also preserve
   `<think>` tags for other models). The response looks like:

       <|channel|>analysis<|message|>... reasoning ...<|end|>
       <|start|>assistant<|channel|>final<|message|>... answer ...<|return|>

   The post-thinking text is everything after the LAST
   `<|channel|>final<|message|>` marker, with any trailing `<|return|>` /
   `<|end|>` stripped.

2. gpt-oss harmony WITH special tokens stripped. The `<|channel|>` and
   `<|message|>` markers vanish, leaving the channel/role names as raw text:

       analysis... reasoning ...assistantfinal... answer ...

   The post-thinking text is everything after the LAST `assistantfinal`.

3. DeepSeek/Qwen `<think>...</think>`: take the segment after the last
   `</think>`; if `<think>` and `</think>` counts disagree across the full
   prompt+response, tag `malformed_thinking`.

If none of these match, tag `no_thinking` and return the response unchanged.

Truncation: if the post-thinking text is longer than `max_response_length`
AND we couldn't parse the thinking cleanly, cut to that length and prefix
the status with `truncated_`. Parsable rows are never truncated.

Adapts the upstream `parse_thinking_steps` from
`big_math_rl/stage1_init_response_sampling/stage1_postprocess_recursive.py`,
which only handled cases (2) and (3); case (1) is needed when special
tokens are kept in detokenization (our default).
"""

from __future__ import annotations

# gpt-oss harmony special-token markers
_HARMONY_FINAL = "<|channel|>final<|message|>"
_HARMONY_ANALYSIS = "<|channel|>analysis<|message|>"
_HARMONY_END_TOKENS = ("<|return|>", "<|end|>", "<|endoftext|>")


def parse_thinking_steps(response: str, prompt: str,
                         max_response_length: int = 16384) -> tuple[str, str]:
    # Case 1: gpt-oss harmony with special tokens kept.
    if _HARMONY_FINAL in response:
        final_part = response.split(_HARMONY_FINAL)[-1]
        for tok in _HARMONY_END_TOKENS:
            if final_part.endswith(tok):
                final_part = final_part[: -len(tok)]
        return final_part.strip(), "parsable_thinking"
    if _HARMONY_ANALYSIS in response:
        # Started reasoning but never reached the final channel — usually a
        # max_tokens cutoff.
        return response, "malformed_thinking"

    # Case 2: gpt-oss harmony with special tokens stripped.
    if response.startswith("analysis") and len(response) > 7 and response[7] != " ":
        if "assistantfinal" in response:
            return response.split("assistantfinal")[-1], "parsable_thinking"
        return response, "malformed_thinking"

    # Case 3: <think>...</think>
    if "<think>" not in (prompt + response):
        return response, "no_thinking"

    non_thinking_response = response.split("</think>")[-1]
    concatenated = prompt + response
    if concatenated.count("<think>") != concatenated.count("</think>"):
        thinking_status = "malformed_thinking"
    else:
        thinking_status = "parsable_thinking"

    if (len(non_thinking_response) > max_response_length
            and thinking_status != "parsable_thinking"):
        non_thinking_response = non_thinking_response[:max_response_length]
        thinking_status = "truncated_" + thinking_status

    return non_thinking_response, thinking_status
