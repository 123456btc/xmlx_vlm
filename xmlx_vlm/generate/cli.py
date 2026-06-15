from __future__ import annotations
import argparse
import codecs
import json
from pathlib import Path
from typing import List, Optional

from ..utils import load, prepare_inputs
from .single import generate, stream_generate
from ..prompt_utils import apply_chat_template
from .. import diffusion_generate

from .types import (
    DEFAULT_MODEL_PATH,
    DEFAULT_IMAGE,
    DEFAULT_AUDIO,
    DEFAULT_VIDEO,
    DEFAULT_PROMPT,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_SEED,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_REPETITION_CONTEXT_SIZE,
    DEFAULT_KV_GROUP_SIZE,
    DEFAULT_KV_QUANT_SCHEME,
    DEFAULT_QUANTIZED_KV_START,
    DEFAULT_PREFILL_STEP_SIZE,
    DEFAULT_THINKING_START_TOKEN,
    DEFAULT_THINKING_END_TOKEN,
)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=DEFAULT_AUDIO,
        help="URL or path of the audio to process.",
    )
    parser.add_argument(
        "--video",
        type=str,
        nargs="+",
        default=DEFAULT_VIDEO,
        help="URL or path of the video to process.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=2.0,
        help="Frames-per-second to sample from --video.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument("--verbose", action="store_false", help="Detailed output.")
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=float,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-quant-scheme",
        type=str,
        choices=("uniform", "turboquant"),
        default=DEFAULT_KV_QUANT_SCHEME,
        help="KV cache quantization backend. Fractional --kv-bits values use "
        "TurboQuant automatically.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=DEFAULT_KV_GROUP_SIZE,
        help="Group size for uniform KV cache quantization.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model from Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model version to use (branch, tag, commit).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Enable activation quantization for QQLinear layers. "
        "Only supported for models quantized with 'nvfp4' or 'mxfp8' modes.",
    )
    parser.add_argument(
        "--processor-kwargs",
        type=json.loads,
        default={},
        help="Extra processor kwargs as JSON. "
        'Example: --processor-kwargs \'{"cropping": false, "max_patches": 3}\'',
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=DEFAULT_PREFILL_STEP_SIZE,
        help="Number of tokens to process per prefill step. "
        "Lower values reduce peak memory usage but may be slower. "
        "Try 512 or 256 if you hit GPU memory errors during prefill.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Speculative drafter path or HF id (e.g. z-lab/Qwen3.5-4B-DFlash).",
    )
    parser.add_argument(
        "--draft-kind",
        type=str,
        default=None,
        choices=["dflash", "mtp"],
        help="Drafter family. Supported: 'dflash' (Qwen3.5 DFlash), "
        "'mtp' (Gemma 4 Multi-Token Prediction / Assistant model). "
        "Default: auto-detected from the drafter's HF model_type.",
    )
    parser.add_argument(
        "--draft-block-size",
        type=int,
        default=None,
        help="Override the drafter's configured block size.",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Enable thinking mode in the chat template (e.g. for Qwen3.5).",
    )
    parser.add_argument(
        "--enable-specprefill",
        action="store_true",
        help="Enable speculative prefill (SpecPrefill) for long prompts.",
    )
    parser.add_argument(
        "--specprefill-draft-model",
        type=str,
        default=None,
        help="Path or repo to the draft model for speculative prefill scoring.",
    )
    parser.add_argument(
        "--specprefill-keep-pct",
        type=float,
        default=0.3,
        help="Percentage of prompt tokens to keep during sparse prefilling.",
    )
    parser.add_argument(
        "--specprefill-chunk-size",
        type=int,
        default=32,
        help="Chunk size for speculative prefill scoring.",
    )
    parser.add_argument(
        "--specprefill-n-lookahead",
        type=int,
        default=8,
        help="Number of lookahead tokens for speculative prefill scoring.",
    )
    parser.add_argument(
        "--specprefill-threshold",
        type=int,
        default=512,
        help="Threshold length of prompt tokens to trigger speculative prefill.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Maximum number of thinking tokens before forcing the end-of-thinking token.",
    )
    parser.add_argument(
        "--thinking-start-token",
        type=str,
        default=DEFAULT_THINKING_START_TOKEN,
        help="Token that marks the start of a thinking block (default: %(default)s).",
    )
    parser.add_argument(
        "--thinking-end-token",
        type=str,
        default=DEFAULT_THINKING_END_TOKEN,
        help="Token that marks the end of a thinking block (default: %(default)s).",
    )

    return parser.parse_args()


def main():
    import sys
    gen_mod = sys.modules.get("xmlx_vlm.generate")
    dyn_parse_arguments = getattr(gen_mod, "parse_arguments", parse_arguments) if gen_mod else parse_arguments
    dyn_load = getattr(gen_mod, "load", load) if gen_mod else load
    dyn_apply_chat_template = getattr(gen_mod, "apply_chat_template", apply_chat_template) if gen_mod else apply_chat_template

    args = dyn_parse_arguments()
    if isinstance(args.image, str):
        args.image = [args.image]
    if isinstance(args.audio, str):
        args.audio = [args.audio]
    if isinstance(args.video, str):
        args.video = [args.video]

    model, processor = dyn_load(
        args.model,
        args.adapter_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
    )
    config = model.config

    draft_model = None
    if args.draft_model is not None:
        from ..speculative.drafters import load_drafter

        print(f"Loading drafter ({args.draft_kind or 'auto'}): {args.draft_model}")
        draft_model, resolved_kind = load_drafter(
            args.draft_model, kind=args.draft_kind
        )
        if args.draft_kind is None:
            print(f"  → auto-detected --draft-kind={resolved_kind!r}.")
        elif resolved_kind != args.draft_kind:
            print(
                f"  → drafter requires --draft-kind={resolved_kind!r}; "
                f"using {resolved_kind!r} instead of {args.draft_kind!r}."
            )
        args.draft_kind = resolved_kind

    prompt = args.prompt

    num_images = len(args.image) if args.image is not None else 0
    num_audios = len(args.audio) if args.audio is not None else 0

    chat_template_kwargs = {"enable_thinking": args.enable_thinking}
    if args.video:
        chat_template_kwargs["video"] = args.video
        chat_template_kwargs["fps"] = args.fps

    prompt = dyn_apply_chat_template(
        processor,
        config,
        prompt,
        num_images=num_images,
        num_audios=num_audios,
        **chat_template_kwargs,
    )

    kwargs = {}

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Add processor kwargs from JSON
    if args.processor_kwargs:
        kwargs.update(args.processor_kwargs)

    # Add thinking kwargs
    kwargs["enable_thinking"] = args.enable_thinking
    if args.thinking_budget is not None:
        kwargs["thinking_budget"] = args.thinking_budget
        kwargs["thinking_end_token"] = args.thinking_end_token
        if args.thinking_start_token is not None:
            kwargs["thinking_start_token"] = args.thinking_start_token

    # Add SpecPrefill kwargs
    if args.enable_specprefill:
        kwargs["enable_specprefill"] = args.enable_specprefill
        kwargs["specprefill_draft_model"] = args.specprefill_draft_model
        kwargs["specprefill_keep_pct"] = args.specprefill_keep_pct
        kwargs["specprefill_chunk_size"] = args.specprefill_chunk_size
        kwargs["specprefill_n_lookahead"] = args.specprefill_n_lookahead
        kwargs["specprefill_threshold"] = args.specprefill_threshold

    if args.chat:
        from ..vision_cache import VisionFeatureCache

        vision_cache = VisionFeatureCache()
        chat = []
        if args.system:
            chat.append({"role": "system", "content": args.system})
        while user := input("User:"):
            chat.append({"role": "user", "content": user})
            prompt = dyn_apply_chat_template(
                processor,
                config,
                chat,
                num_images=num_images,
                num_audios=num_audios,
                **chat_template_kwargs,
            )
            response = ""
            print("Assistant:", end="")
            stream_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                "vision_cache": vision_cache,
                **kwargs,
            }
            if args.resize_shape is not None:
                stream_kwargs["resize_shape"] = args.resize_shape
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size

            for chunk in dyn_stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                **stream_kwargs,
            ):
                response += chunk.text
                print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            print()

    else:
        gen_kwargs = {
            "image": args.image,
            "audio": args.audio,
            "video": args.video,
            "fps": args.fps,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "verbose": args.verbose,
            "max_kv_size": args.max_kv_size,
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "kv_quant_scheme": getattr(
                args, "kv_quant_scheme", DEFAULT_KV_QUANT_SCHEME
            ),
            "quantized_kv_start": args.quantized_kv_start,
            **kwargs,
        }
        if args.resize_shape is not None:
            gen_kwargs["resize_shape"] = args.resize_shape
        if args.prefill_step_size is not None:
            gen_kwargs["prefill_step_size"] = args.prefill_step_size
        if draft_model is not None:
            gen_kwargs["draft_model"] = draft_model
            gen_kwargs["draft_kind"] = args.draft_kind
            if args.draft_block_size is not None:
                gen_kwargs["draft_block_size"] = args.draft_block_size

        dyn_generate = getattr(gen_mod, "generate", generate) if gen_mod else generate
        result = dyn_generate(
            model,
            processor,
            prompt,
            **gen_kwargs,
        )
        if not args.verbose:
            print(result.text)

        if draft_model is not None:
            lens = getattr(draft_model, "accept_lens", None) or []
            if lens:
                mean_accept = round(sum(lens) / len(lens), 2)
                print(
                    f"Speculative decoding: {mean_accept} accepted tokens over {len(lens)} rounds"
                )


if __name__ == "__main__":
    print(
        "Calling `python -m xmlx_vlm.generate ...` directly is deprecated."
        " Use `xmlx_vlm generate` or `python -m xmlx_vlm generate` instead."
    )
    main()
