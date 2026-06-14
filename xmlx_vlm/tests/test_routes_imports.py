"""
Smoke-test: verify every route module can be imported and its router
registered on a bare FastAPI app without a model loaded.
Run from the project root:  python test_routes_imports.py
"""
import sys
import types
import unittest.mock as mock


def _setup_stubs(original_sys_modules):
    """Stub heavy dependencies so route modules can be imported without a model."""
    STUBS = [
        'mlx', 'mlx.core', 'mlx.nn', 'mlx.optimizers',
        'mlx_lm', 'mlx_lm.sample_utils', 'mlx_lm.tokenizer_utils', 'mlx_lm.generate',
        'mlx_lm.models', 'mlx_lm.models.cache',
        'huggingface_hub', 'safetensors', 'safetensors.numpy',
        'transformers', 'PIL', 'PIL.Image',
        'xmlx_vlm.generate', 'xmlx_vlm.sample_utils',
        'xmlx_vlm.tokenizer_utils', 'xmlx_vlm.utils',
        'xmlx_vlm.prompt_utils', 'xmlx_vlm.structured',
        'xmlx_vlm.constrained', 'xmlx_vlm.tool_logits_bias',
        'xmlx_vlm.embedding_engine', 'xmlx_vlm.rerank_engine',
        'xmlx_vlm.metrics', 'xmlx_vlm.optimizations',
        'xmlx_vlm.prompt_warmup', 'xmlx_vlm.vision_cache',
        'xmlx_vlm.mcp', 'xmlx_vlm.memory', 'xmlx_vlm.reasoning',
        'xmlx_vlm.moe_topk', 'xmlx_vlm.patches',
        'xmlx_vlm.tool_parsers', 'xmlx_vlm.apc',
        'xmlx_vlm.version',
        'xmlx_vlm.speculative', 'xmlx_vlm.speculative.drafters',
    ]
    for m in STUBS:
        mod = types.ModuleType(m)
        mod.__spec__ = None
        sys.modules[m] = mod

    # Provide the minimum real attributes on stubs
    mx_stub = sys.modules['mlx.core']
    mx_stub.array = object
    mx_stub.clear_cache = lambda: None
    mx_stub.get_peak_memory = lambda: 0.0
    mx_stub.default_stream = lambda *a, **kw: None
    mx_stub.default_device = lambda: None
    mx_stub.new_thread_local_stream = lambda *a, **kw: mock.MagicMock()
    mx_stub.stream = mock.MagicMock()
    mx_stub.compile = lambda f=None, *a, **kw: (f if f else lambda x: x)

    # transformers imports used by diffusion_generate / model code
    transformers_stub = sys.modules['transformers']
    transformers_stub.PreTrainedTokenizer = mock.MagicMock()
    transformers_stub.AutoTokenizer = mock.MagicMock()
    transformers_stub.AutoProcessor = mock.MagicMock()

    # huggingface_hub imports used by routes.admin
    hf_stub = sys.modules['huggingface_hub']
    hf_stub.scan_cache_dir = mock.MagicMock()

    # mlx.utils is imported by xmlx_vlm.convert
    mlx_utils_stub = types.ModuleType('mlx.utils')
    mlx_utils_stub.tree_map_with_path = lambda f, tree: tree
    mlx_utils_stub.tree_reduce = lambda f, tree: tree
    sys.modules['mlx.utils'] = mlx_utils_stub

    gen_stub = sys.modules['xmlx_vlm.generate']
    gen_stub.DEFAULT_KV_GROUP_SIZE = 64
    gen_stub.DEFAULT_KV_QUANT_SCHEME = 'uniform'
    gen_stub.DEFAULT_MAX_TOKENS = 2048
    gen_stub.DEFAULT_MODEL_PATH = ''
    gen_stub.DEFAULT_PREFILL_STEP_SIZE = 512
    gen_stub.DEFAULT_QUANTIZED_KV_START = 0
    gen_stub.DEFAULT_SEED = None
    gen_stub.DEFAULT_TEMPERATURE = 0.0
    gen_stub.DEFAULT_TOP_P = 1.0
    gen_stub.DEFAULT_THINKING_START_TOKEN = '<think>'
    gen_stub.DEFAULT_THINKING_END_TOKEN = '</think>'
    gen_stub.BatchGenerator = mock.MagicMock()
    gen_stub.BatchResponse = mock.MagicMock()
    gen_stub.BatchStats = mock.MagicMock()
    gen_stub.GenerationResult = mock.MagicMock()
    gen_stub.PromptCacheState = mock.MagicMock()
    gen_stub.batch_generate = mock.MagicMock()
    gen_stub.stream_generate = mock.MagicMock()
    gen_stub.generate = mock.MagicMock()
    gen_stub.normalize_resize_shape = mock.MagicMock()
    gen_stub._dflash_rounds_batch = mock.MagicMock()
    gen_stub._mtp_rounds_batch = mock.MagicMock()
    gen_stub._make_cache = mock.MagicMock()
    gen_stub._merge_prefill_prompt_kwargs = mock.MagicMock()
    gen_stub._apply_rep_penalty = mock.MagicMock()
    gen_stub._dflash_rounds = mock.MagicMock()
    gen_stub._mtp_rounds = mock.MagicMock()

    ver_stub = sys.modules['xmlx_vlm.version']
    ver_stub.__version__ = '0.0.0-test'

    apc_stub = sys.modules['xmlx_vlm.apc']
    apc_stub.APCManager = mock.MagicMock()
    apc_stub.from_env = mock.MagicMock(return_value=None)
    apc_stub.hash_image_payload = mock.MagicMock(return_value=b'')

    mcp_stub = sys.modules['xmlx_vlm.mcp']
    mcp_stub.get_manager = mock.MagicMock()

    memory_stub = sys.modules['xmlx_vlm.memory']
    memory_stub.get_memory_store = mock.MagicMock(return_value=None)

    sys.modules['xmlx_vlm.metrics'].metrics = mock.MagicMock()
    sys.modules['xmlx_vlm.optimizations'].detect_hardware = mock.MagicMock()

    for attr in ['load_warmup_file', 'warm_prompts']:
        setattr(sys.modules['xmlx_vlm.prompt_warmup'], attr, mock.MagicMock())

    sys.modules['xmlx_vlm.vision_cache'].VisionFeatureCache = mock.MagicMock()
    sys.modules['xmlx_vlm.patches'].apply_all_patches = mock.MagicMock()
    sys.modules['xmlx_vlm.moe_topk'].apply_moe_top_k_override = mock.MagicMock()
    sys.modules['xmlx_vlm.tool_parsers']._infer_tool_parser_from_processor = mock.MagicMock()
    sys.modules['xmlx_vlm.tool_parsers'].load_tool_module = mock.MagicMock()
    sys.modules['xmlx_vlm.reasoning'].get_parser = mock.MagicMock()
    sys.modules['xmlx_vlm.sample_utils'].top_p_sampling = mock.MagicMock()
    sys.modules['xmlx_vlm.tokenizer_utils'].make_streaming_detokenizer = mock.MagicMock()
    sys.modules['mlx_lm.generate'].maybe_quantize_kv_cache = mock.MagicMock()

    # mlx_lm.models.cache classes used by xmlx_vlm.models.cache
    lm_cache_stub = sys.modules['mlx_lm.models.cache']
    for cls in ['ArraysCache', 'BatchKVCache', 'BatchRotatingKVCache', 'CacheList',
                'ChunkedKVCache', 'KVCache', 'QuantizedKVCache', 'RotatingKVCache', '_BaseCache']:
        setattr(lm_cache_stub, cls, mock.MagicMock())
    lm_cache_stub.create_attention_mask = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].load = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].prepare_inputs = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].MODEL_CONVERSION_DTYPES = ["float16", "bfloat16", "float32"]
    sys.modules['xmlx_vlm.utils'].create_model_card = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].fetch_from_hub = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].get_model_path = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].save_config = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].save_weights = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].skip_multimodal_module = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].upload_to_hub = mock.MagicMock()
    sys.modules['xmlx_vlm.utils'].process_image = mock.MagicMock()
    sys.modules['xmlx_vlm.prompt_utils'].apply_chat_template = mock.MagicMock()
    sys.modules['xmlx_vlm.prompt_utils'].get_message_json = mock.MagicMock()
    sys.modules['xmlx_vlm.structured'].build_json_schema_logits_processor = mock.MagicMock()
    sys.modules['xmlx_vlm.constrained'].ThinkingAwareLogitsProcessor = mock.MagicMock()
    sys.modules['xmlx_vlm.tool_logits_bias'].ToolLogitsBiasProcessor = mock.MagicMock()
    sys.modules['xmlx_vlm.embedding_engine'].EmbeddingEngine = mock.MagicMock()
    sys.modules['xmlx_vlm.rerank_engine'].RerankEngine = mock.MagicMock()

    return original_sys_modules


def test_routes_import_and_register():
    """Stub heavy deps, import route modules, register routers, then clean up."""
    _original_sys_modules = dict(sys.modules)
    _setup_stubs(_original_sys_modules)

    errors = []

    def try_import(label, fn):
        try:
            fn()
            print(f'  OK  {label}')
        except Exception as e:
            print(f'  !!  {label}: {type(e).__name__}: {e}')
            errors.append(label)

    try_import('xmlx_vlm.config',        lambda: __import__('xmlx_vlm.config'))
    try_import('xmlx_vlm.auth',          lambda: __import__('xmlx_vlm.auth'))
    try_import('xmlx_vlm.engine',        lambda: __import__('xmlx_vlm.engine'))
    try_import('xmlx_vlm.model_store',   lambda: __import__('xmlx_vlm.model_store'))
    try_import('routes.completions',     lambda: __import__('xmlx_vlm.routes.completions'))
    try_import('routes.responses',       lambda: __import__('xmlx_vlm.routes.responses'))
    try_import('routes.anthropic',       lambda: __import__('xmlx_vlm.routes.anthropic'))
    try_import('routes.mcp',             lambda: __import__('xmlx_vlm.routes.mcp'))
    try_import('routes.admin',           lambda: __import__('xmlx_vlm.routes.admin'))
    try_import('xmlx_vlm.app',           lambda: __import__('xmlx_vlm.app'))

    # ── verify router objects are present ───────────────────────────────────────
    from fastapi import FastAPI
    from xmlx_vlm.routes.completions import router as cr
    from xmlx_vlm.routes.responses   import router as rr
    from xmlx_vlm.routes.anthropic   import router as ar
    from xmlx_vlm.routes.mcp         import router as mr
    from xmlx_vlm.routes.admin       import router as dr

    bare_app = FastAPI()
    for name, r in [('completions', cr), ('responses', rr), ('anthropic', ar), ('mcp', mr), ('admin', dr)]:
        try:
            bare_app.include_router(r)
            print(f'  OK  router/{name} registered ({len(r.routes)} routes)')
        except Exception as e:
            print(f'  !!  router/{name}: {e}')
            errors.append(name)

    print()
    if errors:
        print(f'FAILED: {errors}')

    try:
        assert not errors, f"route import/registration failures: {errors}"
    finally:
        # Restore the original module state so stub modules don't leak into
        # other tests when this file is collected as part of a suite.
        sys.modules.clear()
        sys.modules.update(_original_sys_modules)
