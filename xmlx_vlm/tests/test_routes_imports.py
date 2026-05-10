"""
Smoke-test: verify every route module can be imported and its router
registered on a bare FastAPI app without a model loaded.
Run from the project root:  python test_routes_imports.py
"""
import sys, types, unittest.mock as mock

# ── stub all native / heavy deps before any project import ──────────────────
STUBS = [
    'mlx', 'mlx.core', 'mlx.nn', 'mlx.optimizers',
    'mlx_lm', 'mlx_lm.sample_utils', 'mlx_lm.tokenizer_utils',
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
    'xmlx_vlm.server_schemas', 'xmlx_vlm.version',
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
mx_stub.stream = mock.MagicMock()

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
gen_stub.stream_generate = mock.MagicMock()
gen_stub.generate = mock.MagicMock()
gen_stub.normalize_resize_shape = mock.MagicMock()
gen_stub._dflash_rounds_batch = mock.MagicMock()
gen_stub._mtp_rounds_batch = mock.MagicMock()
gen_stub._make_cache = mock.MagicMock()
gen_stub._merge_prefill_prompt_kwargs = mock.MagicMock()

schemas_stub = sys.modules['xmlx_vlm.server_schemas']
for cls in ['ChatRequest','ChatResponse','ChatChoice','ChatStreamChunk',
            'ChatStreamChoice','ChatMessage','ChatLogprobContent','ChatLogprobs',
            'TopLogprob','UsageStats','OpenAIRequest','OpenAIResponse',
            'AnthropicMessageRequest','AnthropicMessageResponse','AnthropicMessageContent','AnthropicUsage',
            'EmbeddingRequest','EmbeddingResponse','EmbeddingData','EmbeddingUsage',
            'RerankRequest','RerankResponse','RerankResult','RerankDocument',
            'MCPExecuteRequest','ModelsResponse','MessageItem','ContentPartOutputText',
            'ResponseCreatedEvent','ResponseInProgressEvent','ResponseCompletedEvent',
            'ResponseOutputItemAddedEvent','ResponseOutputItemDoneEvent',
            'ResponseOutputTextDeltaEvent','ResponseOutputTextDoneEvent',
            'ResponseContentPartAddedEvent','ResponseContentPartDoneEvent']:
    setattr(schemas_stub, cls, mock.MagicMock())
schemas_stub.get_server_max_tokens = lambda: 2048

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
sys.modules['xmlx_vlm.utils'].load = mock.MagicMock()
sys.modules['xmlx_vlm.utils'].prepare_inputs = mock.MagicMock()
sys.modules['xmlx_vlm.prompt_utils'].apply_chat_template = mock.MagicMock()
sys.modules['xmlx_vlm.structured'].build_json_schema_logits_processor = mock.MagicMock()
sys.modules['xmlx_vlm.constrained'].ThinkingAwareLogitsProcessor = mock.MagicMock()
sys.modules['xmlx_vlm.tool_logits_bias'].ToolLogitsBiasProcessor = mock.MagicMock()
sys.modules['xmlx_vlm.embedding_engine'].EmbeddingEngine = mock.MagicMock()
sys.modules['xmlx_vlm.rerank_engine'].RerankEngine = mock.MagicMock()

# ── now import the real modules ──────────────────────────────────────────────
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
for name, r in [('completions',cr),('responses',rr),('anthropic',ar),('mcp',mr),('admin',dr)]:
    try:
        bare_app.include_router(r)
        print(f'  OK  router/{name} registered ({len(r.routes)} routes)')
    except Exception as e:
        print(f'  !!  router/{name}: {e}')
        errors.append(name)

print()
if errors:
    print(f'FAILED: {errors}')
    sys.exit(1)
else:
    print('ALL OK — no missing imports detected')
    sys.exit(0)
