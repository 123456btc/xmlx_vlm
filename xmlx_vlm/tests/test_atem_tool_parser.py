"""Tests for ATEM Tool Parser and Broken-output Auto Recovery."""

import json
import pytest

from xmlx_vlm.tool_parsers import _infer_tool_parser, ToolParserManager
from xmlx_vlm.tool_parsers.atem_tool_parser import AtemToolParser
from xmlx_vlm.tool_parsers.recovery import attempt_recovery, auto_recover_tool_calls


def test_infer_atem_tool_parser():
    template = "Some prompt with <atem:call name=foo> template marker"
    assert _infer_tool_parser(template) == "atem"

    template2 = "Some prompt with atem_tool_call marker"
    assert _infer_tool_parser(template2) == "atem"


def test_atem_parser_registered():
    parser_cls = ToolParserManager.get_tool_parser("atem")
    assert parser_cls is AtemToolParser


def test_atem_call_tag_extraction():
    parser = AtemToolParser(tokenizer=None)
    output = (
        "<atem:deliberation>The user is asking for weather data in Tokyo.</atem:deliberation>\n"
        "<atem:call name=\"get_weather\">{\"location\": \"Tokyo\", \"unit\": \"celsius\"}</atem:call>"
    )
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["function"]["name"] == "get_weather"
    args = json.loads(call["function"]["arguments"])
    assert args["location"] == "Tokyo"
    assert args["unit"] == "celsius"
    # Deliberation should be stripped from final content
    assert "<atem:deliberation>" not in result.content
    assert "Tokyo" not in result.content


def test_atem_tool_call_wrapper():
    parser = AtemToolParser(tokenizer=None)
    output = (
        "I will check the factor metrics.\n"
        "<atem_tool_call>\n"
        "{\"name\": \"calculate_alpha\", \"arguments\": {\"factor\": \"momentum\", \"lookback\": 20}}\n"
        "</atem_tool_call>"
    )
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["function"]["name"] == "calculate_alpha"
    args = json.loads(call["function"]["arguments"])
    assert args["factor"] == "momentum"
    assert args["lookback"] == 20
    assert result.content == "I will check the factor metrics."


def test_atem_channel_format():
    parser = AtemToolParser(tokenizer=None)
    output = (
        "<|channel|>thought Examining portfolio risk constraints <|channel|>"
        "<|channel|>call:rebalance_portfolio\n"
        "{\"target_weights\": {\"BTC\": 0.6, \"ETH\": 0.4}}\n"
        "<|endofcall|>"
    )
    result = parser.extract_tool_calls(output)

    assert result.tools_called is True
    assert len(result.tool_calls) == 1
    call = result.tool_calls[0]
    assert call["function"]["name"] == "rebalance_portfolio"
    args = json.loads(call["function"]["arguments"])
    assert args["target_weights"]["BTC"] == 0.6


def test_auto_recovery_unclosed_xml_and_trailing_comma():
    # Broken model output: unclosed <tool_call> and trailing comma in JSON
    broken_output = "<tool_call>{\"name\": \"execute_trade\", \"arguments\": {\"symbol\": \"BTC\", \"amount\": 1.5,}"
    recovered = attempt_recovery(broken_output)

    assert recovered is not None
    assert len(recovered) == 1
    assert recovered[0]["name"] == "execute_trade"
    assert recovered[0]["arguments"]["symbol"] == "BTC"
    assert recovered[0]["arguments"]["amount"] == 1.5


def test_auto_recovery_unclosed_string():
    # Broken model output: unclosed string at end
    broken_output = "<tool_call>{\"name\": \"search_news\", \"arguments\": {\"query\": \"fed interest rate"
    recovered = auto_recover_tool_calls(broken_output)

    assert recovered is not None
    assert len(recovered) == 1
    assert recovered[0]["name"] == "search_news"
    assert "fed interest rate" in recovered[0]["arguments"]["query"]
