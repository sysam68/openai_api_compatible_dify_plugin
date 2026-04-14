from dify_plugin.entities.model.llm import LLMMode
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    SystemPromptMessage,
    TextPromptMessageContent,
    UserPromptMessage,
)

from models.llm.llm import OpenAILargeLanguageModel


def test_wrap_reasoning_with_new_key_streaming():
    m = OpenAILargeLanguageModel(model_schemas=[])
    is_reasoning = False

    # 1) start reasoning with new key 'reasoning'
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"reasoning": "A"}, is_reasoning)
    assert out == "<think>\nA"
    assert is_reasoning is True

    # 2) continue reasoning
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"reasoning": "B"}, is_reasoning)
    assert out == "B"
    assert is_reasoning is True

    # 3) close reasoning block when normal content arrives
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"content": "Hello"}, is_reasoning)
    assert out == "\n</think>Hello"
    assert is_reasoning is False


def test_wrap_reasoning_with_legacy_key_streaming():
    m = OpenAILargeLanguageModel(model_schemas=[])
    is_reasoning = False

    # 1) start reasoning with legacy key 'reasoning_content'
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"reasoning_content": "X"}, is_reasoning)
    assert out == "<think>\nX"
    assert is_reasoning is True

    # 2) continue reasoning
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"reasoning_content": "Y"}, is_reasoning)
    assert out == "Y"
    assert is_reasoning is True

    # 3) close reasoning block on next plain content
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"content": "Z"}, is_reasoning)
    assert out == "\n</think>Z"
    assert is_reasoning is False


def test_wrap_reasoning_end_without_followup_content():
    m = OpenAILargeLanguageModel(model_schemas=[])
    # reasoning already started
    is_reasoning = True

    # delta without reasoning/content should just close the block
    out, is_reasoning = m._wrap_thinking_by_reasoning_content({}, is_reasoning)
    assert out == "\n</think>"
    assert is_reasoning is False


def test_wrap_plain_content_when_not_in_reasoning():
    m = OpenAILargeLanguageModel(model_schemas=[])
    is_reasoning = False

    out, is_reasoning = m._wrap_thinking_by_reasoning_content({"content": "plain"}, is_reasoning)
    assert out == "plain"
    assert is_reasoning is False


def test_wrap_reasoning_handles_list_content_fragments():
    m = OpenAILargeLanguageModel(model_schemas=[])
    is_reasoning = False

    out, is_reasoning = m._wrap_thinking_by_reasoning_content(
        {"reasoning": [{"type": "text", "text": "A"}]},
        is_reasoning,
    )
    assert out == "<think>\nA"
    assert is_reasoning is True

    out, is_reasoning = m._wrap_thinking_by_reasoning_content(
        {"content": [{"type": "text", "text": "B"}]},
        is_reasoning,
    )
    assert out == "\n</think>B"
    assert is_reasoning is False


def test_prepend_structured_output_prompt_preserves_multimodal_system_prompt():
    system_prompt = SystemPromptMessage(
        content=[TextPromptMessageContent(data="Existing system prompt.")]
    )

    OpenAILargeLanguageModel._prepend_structured_output_prompt(
        system_prompt,
        "Return valid JSON.",
    )

    assert isinstance(system_prompt.content, list)
    assert len(system_prompt.content) == 2
    assert system_prompt.content[0].data == "Return valid JSON.\n\n"
    assert system_prompt.content[1].data == "Existing system prompt."


class _DummyStreamResponse:
    def __init__(self, chunks):
        self._chunks = chunks

    def iter_lines(self, decode_unicode=True, delimiter="\n\n"):
        for chunk in self._chunks:
            yield chunk


class _DummyJSONResponse:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


def test_handle_generate_stream_response_coerces_list_delta_content():
    model = OpenAILargeLanguageModel(model_schemas=[])
    response = _DummyStreamResponse(
        [
            'data: {"choices":[{"delta":{"content":[{"type":"text","text":"Hello"}]}}]}',
            'data: {"choices":[{"delta":{"content":[{"type":"text","text":" world"}]},"finish_reason":"stop"}]}',
        ]
    )

    chunks = list(
        model._handle_generate_stream_response(
            "gpt-4o",
            {"mode": LLMMode.CHAT.value},
            response,
            [UserPromptMessage(content="prompt")],
        )
    )

    assert chunks[0].delta.message.content == "Hello"
    assert chunks[1].delta.message.content == " world"
    assert chunks[-1].delta.finish_reason == "stop"


def test_handle_generate_response_coerces_list_message_content():
    model = OpenAILargeLanguageModel(model_schemas=[])
    response = _DummyJSONResponse(
        {
            "id": "resp_1",
            "model": "gpt-4o",
            "choices": [
                {
                    "message": {
                        "content": [
                            {"type": "text", "text": "Hello"},
                            {"type": "text", "text": " world"},
                        ]
                    }
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 2},
        }
    )

    result = model._handle_generate_response(
        "gpt-4o",
        {"mode": LLMMode.CHAT.value},
        response,
        [UserPromptMessage(content="prompt")],
    )

    assert result.message.content == "Hello world"
