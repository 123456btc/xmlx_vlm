#!/bin/bash
# XMLX VLM Service Manager
# Usage:
#   ./service.sh start [--chat]     Start server (optionally with chat UI)
#   ./service.sh stop               Stop server and chat UI
#   ./service.sh restart [--chat]   Restart server
#   ./service.sh status             Show running status
#
# Configurable via environment variables (or edit defaults below):
#   XMLX_VLM_MODEL       Default model to load
#   XMLX_VLM_PORT        Server port (default: 5118)
#   XMLX_VLM_CHAT_PORT   Chat UI port (default: 5119)
#   XMLX_VLM_API_KEY     API key for auth (default: x123456)
#   XMLX_VLM_ENABLE_THINKING Enable thinking mode (default: true)
#   XMLX_VLM_ARGS        Extra args passed to server
#   XMLX_VLM_WATCHLIST   Comma-separated custom watchlist (e.g. BTC,ETH,SOL)
#   XMLX_VLM_WATCHLIST_SIZE  Number of top volume coins to trade (default: 3)
#
# Direct server options (passed through to server):
#   --draft-model MODEL         Speculative drafter model
#   --draft-kind {dflash,mtp}   Drafter family
#   --kv-bits BITS              KV cache quantization bits
#   --kv-quant-scheme SCHEME    {uniform,turboquant}
#   --thinking / --enable-thinking     Enable thinking mode (default: enabled)
#   --no-thinking / --disable-thinking Disable thinking mode

set -euo pipefail

export NO_PROXY="localhost,127.0.0.1,0.0.0.0,::1"
export no_proxy="localhost,127.0.0.1,0.0.0.0,::1"

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/.logs"

MODEL="${XMLX_VLM_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
DEFAULT_MODEL="mlx-community/Qwen3.8-27B-4bit"
MODEL_SPECIFIED=false
PORT="${XMLX_VLM_PORT:-5118}"
CHAT_PORT="${XMLX_VLM_CHAT_PORT:-5119}"
API_KEY="${XMLX_VLM_API_KEY:-x123456}"
EXTRA_ARGS="${XMLX_VLM_ARGS:-}"

# Default speculative decoding config
DRAFT_MODEL="${XMLX_VLM_DRAFT_MODEL:-}"
DRAFT_KIND="${XMLX_VLM_DRAFT_KIND:-dflash}"

# Default TurboQuant config
KV_BITS="${XMLX_VLM_KV_BITS:-3.5}"
KV_QUANT_SCHEME="${XMLX_VLM_KV_QUANT_SCHEME:-turboquant}"

# Default to launching chat UI alongside the server.
# Set XMLX_VLM_CHAT=false (or use --no-chat) to start server only.
CHAT_ENABLED="${XMLX_VLM_CHAT:-true}"

# Thinking mode (default: true, set XMLX_VLM_ENABLE_THINKING=false or use --no-thinking to disable)
ENABLE_THINKING="${XMLX_VLM_ENABLE_THINKING:-true}"

# Computer Use mode: gui | autonomous | gui_voice | autonomous_voice
COMPUTER_MODE="${XMLX_VLM_COMPUTER_MODE:-autonomous}"

# Strategy engine auto-start (default: false for interactive chat mode, --strategy to enable)
START_STRATEGY="${XMLX_VLM_STRATEGY:-false}"

# Watchlist configuration
WATCHLIST="${XMLX_VLM_WATCHLIST:-}"
WATCHLIST_SIZE="${XMLX_VLM_WATCHLIST_SIZE:-3}"

SERVER_PID_FILE="${PID_DIR}/server.pid"
CHAT_PID_FILE="${PID_DIR}/chat.pid"
STRATEGY_PID_FILE="${PID_DIR}/strategies.pid"
SERVER_LOG="${LOG_DIR}/server.log"
CHAT_LOG="${LOG_DIR}/chat.log"
STRATEGY_LOG="${LOG_DIR}/strategies.log"
STRATEGY_CONFIG="${SCRIPT_DIR}/xmlx_vlm/ai_trader/strategies.json"

# ─── Colors & Formatting ───────────────────────────────────────────────────
if [[ -t 1 ]]; then
    BOLD='\033[1m'
    DIM='\033[2m'
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[0;33m'
    BLUE='\033[0;34m'
    MAGENTA='\033[0;35m'
    CYAN='\033[0;36m'
    WHITE='\033[1;37m'
    NC='\033[0m' # No Color
else
    BOLD=''
    DIM=''
    RED=''
    GREEN=''
    YELLOW=''
    BLUE=''
    MAGENTA=''
    CYAN=''
    WHITE=''
    NC=''
fi

# ─── Helpers ────────────────────────────────────────────────────────────────
ensure_dirs() {
    mkdir -p "${PID_DIR}" "${LOG_DIR}"
}

is_running() {
    local pid_file="$1"
    if [[ -f "$pid_file" ]]; then
        local pid
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

pid_of_port() {
    local port="$1"
    lsof -t -i":${port}" 2>/dev/null || true
}

kill_port() {
    local port="$1"
    local pids
    pids="$(pid_of_port "$port")"
    if [[ -n "$pids" ]]; then
        echo "  Killing process on port ${port}: ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 1
        # Force kill if still alive
        for pid in ${pids}; do
            kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        done
    fi
}

print_service_summary() {
    local title="${1:-XMLX-VLM 服务运行信息 / Service Information}"

    # Query live model name if server is running
    local live_model=""
    if is_running "$SERVER_PID_FILE" || [[ -n "$(pid_of_port "$PORT")" ]]; then
        live_model="$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null | grep -o '"loaded_model":"[^"]*"' | cut -d'"' -f4 || true)"
    fi
    local display_model="${live_model:-$MODEL}"

    local server_pid_val=""
    local server_state="stopped"
    if is_running "$SERVER_PID_FILE"; then
        server_pid_val="$(cat "$SERVER_PID_FILE" 2>/dev/null || true)"
        server_state="running"
    else
        local p_pid
        p_pid="$(pid_of_port "$PORT")"
        if [[ -n "$p_pid" ]]; then
            server_pid_val="${p_pid}"
            server_state="running"
            echo "$p_pid" > "$SERVER_PID_FILE" 2>/dev/null || true
        fi
    fi

    local chat_pid_val=""
    local chat_state="stopped"
    if is_running "$CHAT_PID_FILE"; then
        chat_pid_val="$(cat "$CHAT_PID_FILE" 2>/dev/null || true)"
        chat_state="running"
    else
        local c_pid
        c_pid="$(pid_of_port "$CHAT_PORT")"
        if [[ -n "$c_pid" ]]; then
            chat_pid_val="${c_pid}"
            chat_state="running"
            echo "$c_pid" > "$CHAT_PID_FILE" 2>/dev/null || true
        fi
    fi

    local strategy_pid_val=""
    local strategy_state="stopped"
    if is_running "$STRATEGY_PID_FILE"; then
        strategy_pid_val="$(cat "$STRATEGY_PID_FILE" 2>/dev/null || true)"
        strategy_state="running"
    fi

    printf "\n"
    printf "%b\n" "${CYAN}================================================================================${NC}"
    printf "%b\n" "  ${BOLD}${WHITE}🚀 ${title}${NC}"
    printf "%b\n" "${CYAN}================================================================================${NC}"
    printf "\n"

    # ── 1. Model Inference API ──
    printf "%b\n" "  ${BOLD}${GREEN}🧠 1. 本地大模型推理 API / Local Model Inference API${NC}"
    printf "%b\n" "  ${DIM}────────────────────────────────────────────────────────────────────────────${NC}"
    if [[ "$server_state" == "running" ]]; then
        printf "%b\n" "  • 运行状态 (Status):     ${GREEN}● 运行中 / Running${NC} (PID: ${server_pid_val})"
    else
        printf "%b\n" "  • 运行状态 (Status):     ${RED}○ 已停止 / Stopped${NC}"
    fi
    printf "%b\n" "  • 访问端口 (Port):       ${BOLD}${YELLOW}${PORT}${NC}"
    printf "%b\n" "  • 服务地址 (Base URL):   ${BOLD}${CYAN}http://localhost:${PORT}${NC}  ${DIM}(或 http://127.0.0.1:${PORT})${NC}"
    if [[ -n "$API_KEY" ]]; then
        printf "%b\n" "  • 访问密码 / 密钥 (Key): ${BOLD}${YELLOW}${API_KEY}${NC}"
        printf "%b\n" "  • 认证请求头 (Header):   ${BOLD}Authorization: Bearer ${API_KEY}${NC}"
    else
        printf "%b\n" "  • 访问密码 / 密钥 (Key): ${DIM}无需密码 (未启用 API_KEY)${NC}"
    fi
    printf "%b\n" "  • 当前加载模型 (Model):  ${CYAN}${display_model}${NC}"
    printf "%b\n" "  • 核心协议端点 (API Endpoints):"
    printf "%b\n" "    - OpenAI 聊天接口:     ${DIM}POST${NC} ${BOLD}http://localhost:${PORT}/v1/chat/completions${NC}"
    printf "%b\n" "    - Anthropic 消息接口:  ${DIM}POST${NC} ${BOLD}http://localhost:${PORT}/v1/messages${NC}"
    printf "%b\n" "    - 模型信息查询:        ${DIM}GET${NC}  ${BOLD}http://localhost:${PORT}/v1/models${NC}"
    printf "%b\n" "    - 服务健康检查:        ${DIM}GET${NC}  ${BOLD}http://localhost:${PORT}/health${NC}"
    printf "\n"

    # ── 2. Trading OS Web Console ──
    printf "%b\n" "  ${BOLD}${GREEN}📊 2. Trading OS 量化交易 Web 终端 / Trading OS Web Console${NC}"
    printf "%b\n" "  ${DIM}────────────────────────────────────────────────────────────────────────────${NC}"
    if [[ "$chat_state" == "running" ]]; then
        printf "%b\n" "  • 运行状态 (Status):     ${GREEN}● 运行中 / Running${NC} (PID: ${chat_pid_val})"
    else
        printf "%b\n" "  • 运行状态 (Status):     ${RED}○ 已停止 / Stopped${NC}"
    fi
    printf "%b\n" "  • 访问端口 (Port):       ${BOLD}${YELLOW}${CHAT_PORT}${NC}"
    printf "%b\n" "  • 终端网址 (Web URL):    ${BOLD}${CYAN}http://localhost:${CHAT_PORT}${NC}"
    if [[ -n "$API_KEY" ]]; then
        printf "%b\n" "  • 终端密码 / Token:      ${BOLD}${YELLOW}${API_KEY}${NC} ${DIM}(系统已自动联通配置)${NC}"
    fi
    printf "%b\n" "  • 终端功能:              ${DIM}4 角色多智能体自主盯盘、实时行情看板、策略执行与风控审批${NC}"
    printf "\n"

    # ── 3. AI Strategy Engine (if running or requested) ──
    local sec_idx=3
    if [[ "$strategy_state" == "running" || "${START_STRATEGY:-false}" == "true" ]]; then
        printf "%b\n" "  ${BOLD}${GREEN}⚡ ${sec_idx}. AI 策略引擎 / AI Strategy Engine${NC}"
        printf "%b\n" "  ${DIM}────────────────────────────────────────────────────────────────────────────${NC}"
        if [[ "$strategy_state" == "running" ]]; then
            printf "%b\n" "  • 运行状态 (Status):     ${GREEN}● 运行中 / Running${NC} (PID: ${strategy_pid_val})"
        else
            printf "%b\n" "  • 运行状态 (Status):     ${RED}○ 未运行 / Stopped${NC}"
        fi
        if [[ -n "${WATCHLIST}" ]]; then
            printf "%b\n" "  • 监控币种 (Watchlist):  ${CYAN}${WATCHLIST}${NC}"
        else
            printf "%b\n" "  • 监控币种 (Watchlist):  ${CYAN}Top-${WATCHLIST_SIZE} 成交量主流币 (动态轮巡)${NC}"
        fi
        printf "\n"
        sec_idx=4
    fi

    # ── Quick Commands ──
    printf "%b\n" "  ${BOLD}${GREEN}📋 ${sec_idx}. 常用管理指令 / Quick Commands${NC}"
    printf "%b\n" "  ${DIM}────────────────────────────────────────────────────────────────────────────${NC}"
    printf "%b\n" "  • 查看服务状态:         ${BOLD}./service.sh status${NC}"
    printf "%b\n" "  • 查看模型服务日志:     ${BOLD}./service.sh logs server${NC}"
    printf "%b\n" "  • 查看 Trading OS 日志: ${BOLD}./service.sh logs chat${NC}"
    printf "%b\n" "  • 停止所有服务:         ${BOLD}./service.sh stop${NC}"
    printf "%b\n" "  • 重启所有服务:         ${BOLD}./service.sh restart${NC}"
    printf "%b\n" "${CYAN}================================================================================${NC}"
    printf "\n"
}

# ─── Model Selection ────────────────────────────────────────────────────────
select_model_interactive() {
    # If user explicitly passed --model or set XMLX_VLM_MODEL in env, skip prompt
    if [[ "$MODEL_SPECIFIED" == "true" ]]; then
        return 0
    fi
    # If running interactively attached to a TTY, prompt with the model catalog
    if [[ -t 0 ]]; then
        local chosen
        chosen="$("$VENV_PYTHON" -m xmlx_vlm.model_selector --interactive "${MODEL:-$DEFAULT_MODEL}")"
        if [[ -n "$chosen" ]]; then
            MODEL="$chosen"
        fi
    else
        MODEL="${MODEL:-$DEFAULT_MODEL}"
    fi
}

# ─── Commands ───────────────────────────────────────────────────────────────

# Parse extra server args (anything before --chat or non-recognized flags)
parse_server_opts() {
    SERVER_OPTS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model|-m)
                MODEL="$2"
                MODEL_SPECIFIED=true
                shift 2
                ;;
            --port|-p)
                PORT="$2"
                shift 2
                ;;
            --api-key)
                API_KEY="$2"
                shift 2
                ;;
            --draft-model)
                DRAFT_MODEL="$2"
                shift 2
                ;;
            --draft-kind)
                DRAFT_KIND="$2"
                shift 2
                ;;
            --kv-bits)
                KV_BITS="$2"
                shift 2
                ;;
            --kv-bits-per-layer)
                SERVER_OPTS+=("--kv-bits-per-layer" "$2")
                shift 2
                ;;
            --kv-quant-scheme)
                KV_QUANT_SCHEME="$2"
                shift 2
                ;;
            --max-num-seqs)
                SERVER_OPTS+=("--max-num-seqs" "$2")
                shift 2
                ;;
            --chat|--no-chat)
                shift
                ;;
            --thinking|--enable-thinking)
                ENABLE_THINKING=true
                shift
                ;;
            --no-thinking|--disable-thinking|--no-enable-thinking)
                ENABLE_THINKING=false
                shift
                ;;
            --strategy)
                START_STRATEGY=true
                shift
                ;;
            --no-strategy)
                START_STRATEGY=false
                shift
                ;;
            *)
                # Forward other flags (e.g. --enable-tool-logits-bias, etc.)
                SERVER_OPTS+=("$1")
                shift
                ;;
        esac
    done
}

ensure_model_downloaded() {
    local target_model="$1"
    if [[ -d "$target_model" || -f "$target_model" ]]; then
        return 0
    fi
    echo "==> Verifying / downloading model weights for: ${target_model}..."
    echo "    (If not locally cached, full interactive download progress will be shown below)"
    "$VENV_PYTHON" -c "
import sys
from xmlx_vlm.utils import get_model_path
try:
    path = get_model_path(sys.argv[1])
    print(f'==> Model ready on disk: {path}')
except KeyboardInterrupt:
    print('\n[!] Download interrupted by user.')
    sys.exit(130)
except Exception as e:
    print(f'[!] Failed to download model: {e}', file=sys.stderr)
    sys.exit(1)
" "$target_model"
}

cmd_pull() {
    local target_model="${1:-$MODEL}"
    echo "==> Pulling model weights for: ${target_model}"
    "$VENV_PYTHON" -c "
import sys
from xmlx_vlm.utils import get_model_path
try:
    path = get_model_path(sys.argv[1])
    print(f'✓ Model successfully cached at: {path}')
except KeyboardInterrupt:
    print('\n[!] Download interrupted by user.')
    sys.exit(130)
except Exception as e:
    print(f'[!] Failed to download model: {e}', file=sys.stderr)
    sys.exit(1)
" "$target_model"
}

cmd_start() {
    local with_chat=false
    if [[ "${CHAT_ENABLED}" != "false" && "${CHAT_ENABLED}" != "0" ]]; then
        with_chat=true
    fi
    for arg in "$@"; do
        if [[ "$arg" == "--chat" ]]; then
            with_chat=true
        fi
        if [[ "$arg" == "--no-chat" ]]; then
            with_chat=false
        fi
    done

    parse_server_opts "$@"
    select_model_interactive
    ensure_dirs

    # ── Ensure Model is Downloaded (Interactive Foreground Phase) ──
    ensure_model_downloaded "$MODEL"

    # ── Start Server ──
    if is_running "$SERVER_PID_FILE"; then
        echo "Server already running (PID: $(cat "$SERVER_PID_FILE"))"
    else
        echo "Starting XMLX VLM server..."
        echo "  Model: ${MODEL}"
        echo "  Port:  ${PORT}"
        [[ -n "$API_KEY" ]] && echo "  Auth:  enabled"
        if [[ "$ENABLE_THINKING" == "true" || "$ENABLE_THINKING" == "1" ]]; then
            echo "  Thinking: enabled (default)"
        else
            echo "  Thinking: disabled"
        fi
        [[ -n "$DRAFT_MODEL" ]] && echo "  Draft: ${DRAFT_MODEL} (${DRAFT_KIND})"
        [[ -n "$KV_BITS" ]] && echo "  KV:    ${KV_BITS} bits (${KV_QUANT_SCHEME})"
        [[ ${#SERVER_OPTS[@]} -gt 0 ]] && echo "  Extra: ${SERVER_OPTS[*]}"
        [[ -n "$EXTRA_ARGS" ]] && echo "  Env:   ${EXTRA_ARGS}"

        local server_args=(
            -m xmlx_vlm.server
            --model "$MODEL"
            --port "$PORT"
        )
        [[ -n "$API_KEY" ]] && server_args+=(--api-key "$API_KEY")
        # Thinking mode (user can override via --thinking / --no-thinking or SERVER_OPTS)
        local has_thinking_opt=false
        if [[ ${#SERVER_OPTS[@]} -gt 0 ]]; then
            for opt in "${SERVER_OPTS[@]}"; do
                [[ "$opt" == "--enable-thinking" || "$opt" == "--no-enable-thinking" || "$opt" == "--no-thinking" || "$opt" == "--disable-thinking" ]] && has_thinking_opt=true
            done
        fi
        if [[ "$has_thinking_opt" == false ]]; then
            if [[ "$ENABLE_THINKING" == "true" || "$ENABLE_THINKING" == "1" ]]; then
                server_args+=(--enable-thinking)
            else
                server_args+=(--no-enable-thinking)
            fi
        fi
        # Default speculative decoding (user can override via SERVER_OPTS)
        [[ -n "$DRAFT_MODEL" ]] && server_args+=(--draft-model "$DRAFT_MODEL")
        [[ -n "$DRAFT_KIND" ]] && server_args+=(--draft-kind "$DRAFT_KIND")
        # Default TurboQuant KV Cache quantization (user can override via SERVER_OPTS)
        local has_kv_bits=false
        local has_kv_scheme=false
        if [[ ${#SERVER_OPTS[@]} -gt 0 ]]; then
            for opt in "${SERVER_OPTS[@]}"; do
                [[ "$opt" == "--kv-bits" ]] && has_kv_bits=true
                [[ "$opt" == "--kv-quant-scheme" ]] && has_kv_scheme=true
            done
        fi
        if [[ "$has_kv_bits" == false ]]; then
            [[ -n "$KV_BITS" ]] && server_args+=(--kv-bits "$KV_BITS")
        fi
        if [[ "$has_kv_scheme" == false ]]; then
            [[ -n "$KV_QUANT_SCHEME" ]] && server_args+=(--kv-quant-scheme "$KV_QUANT_SCHEME")
        fi
        [[ ${#SERVER_OPTS[@]} -gt 0 ]] && server_args+=("${SERVER_OPTS[@]}")
        [[ -n "$EXTRA_ARGS" ]] && read -ra extra <<< "$EXTRA_ARGS" && server_args+=("${extra[@]}")

        # Clean stale port
        kill_port "$PORT" 2>/dev/null || true

        nohup "$VENV_PYTHON" "${server_args[@]}" > "$SERVER_LOG" 2>&1 &
        local server_pid=$!
        echo "$server_pid" > "$SERVER_PID_FILE"

        # Wait for health check (model is already on disk, only loading weights into RAM)
        local start_timeout="${XMLX_VLM_START_TIMEOUT:-90}"
        local waited=0
        while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
            if ! kill -0 "$server_pid" 2>/dev/null; then
                echo "ERROR: Server process exited unexpectedly. Check ${SERVER_LOG}"
                rm -f "$SERVER_PID_FILE"
                return 1
            fi
            sleep 1
            ((waited++))
            if [[ $waited -ge $start_timeout ]]; then
                echo "ERROR: Server failed to start within ${start_timeout}s. Check ${SERVER_LOG}"
                rm -f "$SERVER_PID_FILE"
                return 1
            fi
        done
        echo "Server ready at http://localhost:${PORT} (PID: ${server_pid})"
    fi

    # ── Start Chat UI ──
    if [[ "$with_chat" == true ]]; then
        if is_running "$CHAT_PID_FILE"; then
            echo "Chat UI already running (PID: $(cat "$CHAT_PID_FILE"))"
        else
            echo "Starting Chat UI..."
            echo "  Port: ${CHAT_PORT}"

            local chat_args=(
                -m xmlx_vlm.ai_trader.web_server
                --server-url "http://localhost:${PORT}"
                --port "$CHAT_PORT"
            )
            [[ -n "$API_KEY" ]] && chat_args+=(--api-key "$API_KEY")
            [[ -n "$MODEL" ]] && chat_args+=(--model "$MODEL")

            # Clean stale port
            kill_port "$CHAT_PORT" 2>/dev/null || true

            nohup "$VENV_PYTHON" "${chat_args[@]}" > "$CHAT_LOG" 2>&1 &
            local chat_pid=$!
            echo "$chat_pid" > "$CHAT_PID_FILE"

            # Wait a bit for Fastapi to bind
            sleep 3
            if ! curl -sf "http://localhost:${CHAT_PORT}" >/dev/null 2>&1; then
                echo "WARNING: Chat UI may still be loading..."
            fi
            echo "Chat UI ready at http://localhost:${CHAT_PORT} (PID: ${chat_pid})"
        fi
    fi

    # ── Start AI Strategy Engine ──
    if [[ "${START_STRATEGY:-false}" == "true" ]]; then
        if is_running "$STRATEGY_PID_FILE"; then
            echo "AI Strategy Engine already running (PID: $(cat "$STRATEGY_PID_FILE"))"
        else
            export XMLX_VLM_WATCHLIST="${WATCHLIST}"
            export XMLX_VLM_WATCHLIST_SIZE="${WATCHLIST_SIZE}"
            if [[ -n "${WATCHLIST}" ]]; then
                echo "Starting AI Strategy Engine (Custom Watchlist Mode: ${WATCHLIST})..."
            else
                echo "Starting AI Strategy Engine (Top-${WATCHLIST_SIZE} Watchlist Dynamic Mode)..."
            fi
            # Start strategies CLI auto-start in the background
            nohup "$VENV_PYTHON" -u -m xmlx_vlm.ai_trader.cli \
                --server-url "http://localhost:${PORT}" \
                --api-key "$API_KEY" \
                --auto-start > "$STRATEGY_LOG" 2>&1 &
            local strategy_pid=$!
            echo "$strategy_pid" > "$STRATEGY_PID_FILE"
            echo "AI Strategy Engine ready (PID: ${strategy_pid})"
        fi
    fi

    # ── Summary & Access Info ──
    print_service_summary "XMLX-VLM 服务已就绪 / Service Ready"
}

cmd_stop() {
    ensure_dirs
    local killed=false

    # ── Stop Server ──
    if is_running "$SERVER_PID_FILE"; then
        local pid
        pid="$(cat "$SERVER_PID_FILE")"
        echo "Stopping server (PID: ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$SERVER_PID_FILE"
        killed=true
    fi
    # Port-level兜底: PID file may be stale or child proc inherited the socket
    local port_pid
    port_pid="$(pid_of_port "$PORT")"
    if [[ -n "$port_pid" ]]; then
        echo "Stopping orphaned server on port ${PORT} (PID: ${port_pid})..."
        kill "$port_pid" 2>/dev/null || true
        sleep 1
        kill -0 "$port_pid" 2>/dev/null && kill -9 "$port_pid" 2>/dev/null || true
        killed=true
    fi

    # ── Stop Chat UI ──
    if is_running "$CHAT_PID_FILE"; then
        local pid
        pid="$(cat "$CHAT_PID_FILE")"
        echo "Stopping chat UI (PID: ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$CHAT_PID_FILE"
        killed=true
    fi
    local chat_pid
    chat_pid="$(pid_of_port "$CHAT_PORT")"
    if [[ -n "$chat_pid" ]]; then
        echo "Stopping orphaned chat UI on port ${CHAT_PORT} (PID: ${chat_pid})..."
        kill "$chat_pid" 2>/dev/null || true
        sleep 1
        kill -0 "$chat_pid" 2>/dev/null && kill -9 "$chat_pid" 2>/dev/null || true
        killed=true
    fi

    # ── Stop AI Strategy Engine ──
    if is_running "$STRATEGY_PID_FILE"; then
        local pid
        pid="$(cat "$STRATEGY_PID_FILE")"
        echo "Stopping AI Strategy Engine (PID: ${pid})..."
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -0 "$pid" 2>/dev/null && kill -9 "$pid" 2>/dev/null || true
        rm -f "$STRATEGY_PID_FILE"
        killed=true
    fi

    if [[ "$killed" == true ]]; then
        echo "Stopped."
    else
        echo "Nothing was running."
    fi
}

cmd_restart() {
    cmd_stop
    sleep 2
    cmd_start "$@"
}

cmd_status() {
    print_service_summary "XMLX-VLM 服务运行状态 / Service Status"
}

cmd_logs() {
    local target="${1:-server}"
    if [[ "$target" == "chat" ]]; then
        tail -f "$CHAT_LOG"
    elif [[ "$target" == "strategy" ]]; then
        tail -f "$STRATEGY_LOG"
    else
        tail -f "$SERVER_LOG"
    fi
}

cmd_computer() {
    local mode="${1:-${COMPUTER_MODE}}"
    case "$mode" in
        gui|autonomous|gui_voice|autonomous_voice)
            ;;
        *)
            echo "Unknown computer mode: $mode"
            echo "Valid modes: gui, autonomous, gui_voice, autonomous_voice"
            exit 1
            ;;
    esac

    echo "Starting XMLX-VLM Computer Use ($mode)..."
    echo "This loads its own GUI agent models and controls your mouse/keyboard."
    echo "Make sure Screen Recording and Accessibility permissions are granted."
    echo ""

    # Run in foreground because computer_use agents are interactive CLIs.
    cd "$SCRIPT_DIR"
    exec "${SCRIPT_DIR}/.venv/bin/xmlx_vlm.computer_use.$mode"
}

# ─── Main ───────────────────────────────────────────────────────────────────

cmd="${1:-}"
shift 2>/dev/null || true

case "$cmd" in
    ""|start)
        cmd_start "$@"
        ;;
    models|list-models|select-model)
        "$VENV_PYTHON" -m xmlx_vlm.model_selector "$@"
        ;;
    pull)
        cmd_pull "$@"
        ;;
    stop)
        cmd_stop
        ;;
    restart)
        cmd_restart "$@"
        ;;
    status)
        cmd_status
        ;;
    logs)
        cmd_logs "$@"
        ;;
    computer)
        cmd_computer "$@"
        ;;
    *)
        cat <<EOF
XMLX VLM Service Manager

Usage:
  $(basename "$0") [start] [--no-chat] [SERVER_OPTS]   Start server with interactive model menu & chat UI
  $(basename "$0") models                                List / select available models interactively
  $(basename "$0") pull [MODEL]                          Pre-download model weights interactively
  $(basename "$0") stop                                   Stop server and chat UI
  $(basename "$0") restart [--no-chat] [SERVER_OPTS]    Restart server
  $(basename "$0") status                                 Show running status
  $(basename "$0") logs [server|chat]                     Tail logs
  $(basename "$0") computer [gui|autonomous|gui_voice|autonomous_voice]
                                                          Launch Computer Use agent (foreground)

Server Options (override defaults passed to xmlx_vlm.server):
  --draft-model MODEL         Speculative drafter (default: ${DRAFT_MODEL})
  --draft-kind {dflash,mtp}   Drafter family (default: ${DRAFT_KIND})
  --kv-bits BITS              KV cache quantization bits (e.g. 3.5, 8)
  --kv-quant-scheme SCHEME    {uniform, turboquant}

Environment:
  XMLX_VLM_MODEL                 Model to load (default: ${MODEL})
  XMLX_VLM_PORT                  Server port (default: ${PORT})
  XMLX_VLM_CHAT_PORT             Chat UI port (default: ${CHAT_PORT})
  XMLX_VLM_API_KEY               API key for auth
  XMLX_VLM_ARGS                  Extra server args (e.g. "--enable-thinking --moe-top-k 4")
  XMLX_VLM_DRAFT_MODEL           Speculative drafter (default: ${DRAFT_MODEL}; set empty to disable)
  XMLX_VLM_DRAFT_KIND            Drafter family (default: ${DRAFT_KIND})
  XMLX_VLM_CHAT                  Launch chat UI (default: true; set "false" to disable)
  XMLX_VLM_COMPUTER_MODE             Default Computer Use mode (default: ${COMPUTER_MODE})
  XMLX_VLM_COMPUTER_GUI_MODEL        Override GUI agent model (default: ${XMLX_VLM_MODEL:-diffusiongemma})
  XMLX_VLM_COMPUTER_PLANNER_MODEL    Override planner model for autonomous mode (default: ${XMLX_VLM_MODEL:-diffusiongemma})
  XMLX_VLM_COMPUTER_WHISPER_MODEL    Override whisper model for voice modes (default: whisper-large-v3-turbo)

Examples:
  # Default start — server + chat UI
  ./$(basename "$0") start

  # Start server only
  ./$(basename "$0") start --no-chat

  # Start autonomous computer use agent
  ./$(basename "$0") computer
  ./$(basename "$0") computer autonomous

  # Start voice-controlled computer use agent
  ./$(basename "$0") computer autonomous_voice

  # Custom auth + KV quantization
  XMLX_VLM_API_KEY=mykey ./$(basename "$0") start --kv-bits 3.5 --kv-quant-scheme turboquant

  # Disable speculative decoding entirely
  XMLX_VLM_DRAFT_MODEL="" XMLX_VLM_DRAFT_KIND="" ./$(basename "$0") start
EOF
        exit 1
        ;;
esac
