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
#   XMLX_VLM_ARGS        Extra args passed to server (e.g. --enable-thinking)
#   XMLX_VLM_WATCHLIST   Comma-separated custom watchlist (e.g. BTC,ETH,SOL)
#   XMLX_VLM_WATCHLIST_SIZE  Number of top volume coins to trade (default: 3)
#
# Direct server options (passed through to server):
#   --draft-model MODEL         Speculative drafter model
#   --draft-kind {dflash,mtp}   Drafter family
#   --kv-bits BITS              KV cache quantization bits
#   --kv-quant-scheme SCHEME    {uniform,turboquant}

set -euo pipefail

# ─── Configuration ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PYTHON="${SCRIPT_DIR}/.venv/bin/python3"
PID_DIR="${SCRIPT_DIR}/.pids"
LOG_DIR="${SCRIPT_DIR}/.logs"

MODEL="${XMLX_VLM_MODEL:-mlx-community/diffusiongemma-26B-A4B-it-4bit}"
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

# Computer Use mode: gui | autonomous | gui_voice | autonomous_voice
COMPUTER_MODE="${XMLX_VLM_COMPUTER_MODE:-autonomous}"

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

# ─── Commands ───────────────────────────────────────────────────────────────

# Parse extra server args (anything before --chat or non-recognized flags)
parse_server_opts() {
    SERVER_OPTS=()
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --draft-model)
                SERVER_OPTS+=("--draft-model" "$2")
                shift 2
                ;;
            --draft-kind)
                SERVER_OPTS+=("--draft-kind" "$2")
                shift 2
                ;;
            --kv-bits)
                SERVER_OPTS+=("--kv-bits" "$2")
                shift 2
                ;;
            --kv-bits-per-layer)
                SERVER_OPTS+=("--kv-bits-per-layer" "$2")
                shift 2
                ;;
            --kv-quant-scheme)
                SERVER_OPTS+=("--kv-quant-scheme" "$2")
                shift 2
                ;;
            --max-num-seqs)
                SERVER_OPTS+=("--max-num-seqs" "$2")
                shift 2
                ;;
            --chat)
                shift
                ;;
            *)
                # Unknown arg — could be extra env args, ignore for now
                shift
                ;;
        esac
    done
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
    ensure_dirs

    # ── Start Server ──
    if is_running "$SERVER_PID_FILE"; then
        echo "Server already running (PID: $(cat "$SERVER_PID_FILE"))"
    else
        echo "Starting XMLX VLM server..."
        echo "  Model: ${MODEL}"
        echo "  Port:  ${PORT}"
        [[ -n "$API_KEY" ]] && echo "  Auth:  enabled"
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

        # Wait for health check
        local waited=0
        while ! curl -sf "http://localhost:${PORT}/health" >/dev/null 2>&1; do
            sleep 1
            ((waited++))
            if [[ $waited -ge 60 ]]; then
                echo "ERROR: Server failed to start within 60s. Check ${SERVER_LOG}"
                rm -f "$SERVER_PID_FILE"
                return 1
            fi
            # Check if process died
            if ! kill -0 "$server_pid" 2>/dev/null; then
                echo "ERROR: Server process exited unexpectedly. Check ${SERVER_LOG}"
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
    local server_status="stopped"
    local server_pid=""
    local server_model=""

    if is_running "$SERVER_PID_FILE"; then
        server_pid="$(cat "$SERVER_PID_FILE")"
        server_status="running"
        server_model="$(curl -sf "http://localhost:${PORT}/health" 2>/dev/null | grep -o '"loaded_model":"[^"]*"' | cut -d'"' -f4 || echo "unknown")"
    else
        local port_pid
        port_pid="$(pid_of_port "$PORT")"
        if [[ -n "$port_pid" ]]; then
            server_pid="${port_pid}"
            server_status="running (orphan)"
        fi
    fi

    local chat_status="stopped"
    local chat_pid=""
    if is_running "$CHAT_PID_FILE"; then
        chat_pid="$(cat "$CHAT_PID_FILE")"
        chat_status="running"
    else
        local port_pid
        port_pid="$(pid_of_port "$CHAT_PORT")"
        if [[ -n "$port_pid" ]]; then
            chat_pid="${port_pid}"
            chat_status="running (orphan)"
        fi
    fi

    local strategy_status="stopped"
    local strategy_pid=""
    if is_running "$STRATEGY_PID_FILE"; then
        strategy_pid="$(cat "$STRATEGY_PID_FILE")"
        strategy_status="running"
    fi

    echo "╔══════════════════════════════════════════╗"
    echo "║        XMLX VLM Service Status           ║"
    echo "╠══════════════════════════════════════════╣"
    printf "║  Server:   %-31s ║\n" "${server_status}"
    [[ -n "$server_pid" ]] && printf "║  PID:      %-31s ║\n" "${server_pid}"
    [[ -n "$server_model" ]] && printf "║  Model:    %-31s ║\n" "${server_model}"
    printf "║  Port:     %-31s ║\n" "${PORT}"
    printf "║  Chat:     %-31s ║\n" "${chat_status}"
    [[ -n "$chat_pid" ]] && printf "║  PID:      %-31s ║\n" "${chat_pid}"
    printf "║  Port:     %-31s ║\n" "${CHAT_PORT}"
    printf "║  Strategy: %-29s ║\n" "${strategy_status}"
    [[ -n "$strategy_pid" ]] && printf "║  PID:      %-29s ║\n" "${strategy_pid}"
    echo "╚══════════════════════════════════════════╝"
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
    start)
        cmd_start "$@"
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
  $(basename "$0") start [--no-chat] [SERVER_OPTS]      Start server with chat UI (default)
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
