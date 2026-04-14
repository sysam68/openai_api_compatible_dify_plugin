from dify_plugin import Plugin, DifyPluginEnv
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel


def _coerce_content_piece(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", item.get("data", ""))))
                elif "content" in item:
                    parts.append(_coerce_content_piece(item.get("content")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(value, dict):
        if value.get("type") == "text":
            return str(value.get("text", value.get("data", "")))
        if "content" in value:
            return _coerce_content_piece(value.get("content"))
    return str(value)


def _patched_wrap_thinking_by_reasoning_content(self, delta: dict, is_reasoning: bool):
    content = _coerce_content_piece(delta.get("content"))
    reasoning_content = _coerce_content_piece(delta.get("reasoning") or delta.get("reasoning_content"))
    output = content
    if reasoning_content:
        if not is_reasoning:
            output = "<think>\n" + reasoning_content
            is_reasoning = True
        else:
            output = reasoning_content
    elif is_reasoning:
        is_reasoning = False
        output = "\n</think>"
        if content:
            output += content

    return output, is_reasoning


LargeLanguageModel._wrap_thinking_by_reasoning_content = _patched_wrap_thinking_by_reasoning_content

plugin = Plugin(DifyPluginEnv())

if __name__ == "__main__":
    plugin.run()
