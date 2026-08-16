#!/bin/bash
# XMLX-VLM One-Click Installer for macOS (Apple Silicon)
# 
# Assumptions:
#   - Fresh Mac or new user (no dev tools installed)
#   - Apple Silicon (M1/M2/M3/M4/M5/...)
#   - Internet connection
#
# What this does:
#   1. Checks macOS + Apple Silicon
#   2. Installs Xcode Command Line Tools (if missing)
#   3. Installs Homebrew (if missing)
#   4. Installs Python 3.14 (if < 3.10)
#   5. Installs uv (fast Python package manager)
#   6. Clones xmlx_vlm repo (if not already inside it)
#   7. Creates virtual environment + installs all dependencies
#   8. Downloads default model (optional)
#   9. Starts the server with chat UI
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/123456btc/xmlx_vlm/master/install.sh | bash
#   OR save and run: ./install.sh

set -euo pipefail

# ─── Colors ─────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── 1. Platform Check ──────────────────────────────────────────────────────
info "Checking platform..."

if [[ "$(uname -s)" != "Darwin" ]]; then
    err "This installer only supports macOS. Detected: $(uname -s)"
    exit 1
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "arm64" ]]; then
    err "XMLX-VLM requires Apple Silicon (arm64). Detected: $ARCH"
    err "Intel Macs are not supported because MLX requires Apple Silicon GPU."
    err "M1/M2/M3/M4/M5 and future Apple Silicon are all supported."
    exit 1
fi
ok "macOS Apple Silicon detected ($ARCH)"

# ─── 2. Xcode Command Line Tools ────────────────────────────────────────────
if ! xcode-select -p &>/dev/null; then
    info "Xcode Command Line Tools not found. Installing..."
    info "(This may take 2-5 minutes. A popup may appear asking for permission.)"
    xcode-select --install 2>/dev/null || true
    # Wait until installed
    until xcode-select -p &>/dev/null; do
        sleep 5
        info "Waiting for Xcode Command Line Tools installation..."
    done
    ok "Xcode Command Line Tools installed"
else
    ok "Xcode Command Line Tools already installed"
fi

# ─── 3. Homebrew ────────────────────────────────────────────────────────────
if ! command -v brew &>/dev/null; then
    info "Homebrew not found. Installing..."
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    # Add brew to PATH for this session
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    elif [[ -f /usr/local/bin/brew ]]; then
        eval "$(/usr/local/bin/brew shellenv)"
    fi
    ok "Homebrew installed"
else
    ok "Homebrew already installed"
    # Ensure brew is in PATH
    if [[ -f /opt/homebrew/bin/brew ]]; then
        eval "$(/opt/homebrew/bin/brew shellenv)"
    fi
fi

# ─── 4. Python 3.10+ (prefer 3.14, the project default) ─────────────────────
info "Checking Python version..."

PYTHON_CMD=""
for cmd in python3.14 python3.13 python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" &>/dev/null; then
        VER="$($cmd --version 2>&1 | awk '{print $2}')"
        MAJOR="$(echo "$VER" | cut -d. -f1)"
        MINOR="$(echo "$VER" | cut -d. -f2)"
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]]; then
            PYTHON_CMD="$cmd"
            ok "Python $VER found at $(command -v "$cmd")"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    info "Python 3.10+ not found. Installing Python 3.14 via Homebrew..."
    brew install python@3.14
    if [[ -f /opt/homebrew/bin/python3.14 ]]; then
        PYTHON_CMD="/opt/homebrew/bin/python3.14"
    elif [[ -f /usr/local/bin/python3.14 ]]; then
        PYTHON_CMD="/usr/local/bin/python3.14"
    else
        err "Failed to install Python 3.14"
        exit 1
    fi
    ok "Python 3.14 installed"
fi

# Ensure pip is available
if ! "$PYTHON_CMD" -m pip --version &>/dev/null; then
    info "Installing pip..."
    "$PYTHON_CMD" -m ensurepip --upgrade 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | "$PYTHON_CMD"
    ok "pip installed"
fi

# ─── 5. uv (fast package manager) ───────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    info "Installing uv (fast Python package manager)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Add uv to PATH for this session
    if [[ -f "$HOME/.local/bin/uv" ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    elif [[ -f /opt/homebrew/bin/uv ]]; then
        export PATH="/opt/homebrew/bin:$PATH"
    fi
    ok "uv installed"
else
    ok "uv already installed"
fi

# Verify uv is available
if ! command -v uv &>/dev/null; then
    err "uv installation failed or not in PATH"
    exit 1
fi

# ─── 6. Clone or enter repo ─────────────────────────────────────────────────
SCRIPT_DIR=""
if [[ -f "pyproject.toml" && -d "xmlx_vlm" ]]; then
    SCRIPT_DIR="$(pwd)"
    info "Already inside xmlx_vlm repository"
else
    REPO_URL="https://github.com/123456btc/xmlx_vlm.git"
    INSTALL_DIR="${HOME}/xmlx_vlm"
    info "Cloning xmlx_vlm repository to $INSTALL_DIR..."
    if [[ -d "$INSTALL_DIR" ]]; then
        warn "Directory $INSTALL_DIR already exists. Pulling latest changes..."
        cd "$INSTALL_DIR"
        git pull origin master
    else
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    SCRIPT_DIR="$INSTALL_DIR"
    ok "Repository ready at $SCRIPT_DIR"
fi

cd "$SCRIPT_DIR"

# ─── 7. Create requirements.txt if missing ──────────────────────────────────
if [[ ! -f "requirements.txt" ]]; then
    info "Creating requirements.txt (project dependency list)..."
    cat > requirements.txt <<'REQEOF'
mlx>=0.20.0
mlx-lm
numpy
pillow
transformers>=4.40.0
torch>=2.0.0
torchvision
fastapi>=0.110.0
uvicorn>=0.29.0
pydantic>=2.0.0
requests
huggingface-hub>=0.22.0
datasets
tqdm
rich
tabulate
scipy
scikit-learn
opencv-python
gradio>=5.19.0
sentencepiece
protobuf
REQEOF
    ok "requirements.txt created"
fi

# ─── 8. Create virtual environment and install ──────────────────────────────
info "Creating Python virtual environment with uv..."

if [[ ! -d ".venv" ]]; then
    uv venv --python "$PYTHON_CMD"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# uv automatically discovers .venv in the current directory, so activation is
# not required. We keep PATH for the uv binary installed earlier.
export PATH="$(pwd)/.venv/bin:${PATH}"

# Install project + all requirements in one pass. uv resolves and downloads the
# dependency graph only once, which is significantly faster than pip.
info "Installing project dependencies with uv (this may take 5-10 minutes on first run)..."
uv pip install -r requirements.txt -e "."
ok "All dependencies installed"

# ─── 9. Environment defaults ────────────────────────────────────────────────
info "Setting default environment variables..."

export XMLX_VLM_API_KEY="${XMLX_VLM_API_KEY:-x123456}"
export XMLX_VLM_MODEL="${XMLX_VLM_MODEL:-mlx-community/Qwen3.8-27B-4bit}"
export XMLX_VLM_PORT="${XMLX_VLM_PORT:-5118}"
export XMLX_VLM_CHAT_PORT="${XMLX_VLM_CHAT_PORT:-5119}"

# Persist to shell profile for future sessions
SHELL_PROFILE=""
if [[ "$SHELL" == */zsh ]]; then
    SHELL_PROFILE="$HOME/.zshrc"
else
    SHELL_PROFILE="$HOME/.bash_profile"
fi

if [[ -n "$SHELL_PROFILE" && -f "$SHELL_PROFILE" ]]; then
    if ! grep -q "XMLX_VLM_API_KEY" "$SHELL_PROFILE" 2>/dev/null; then
        cat >> "$SHELL_PROFILE" <<ENVEOF

# XMLX-VLM defaults
export XMLX_VLM_API_KEY="x123456"
export XMLX_VLM_MODEL="mlx-community/Qwen3.8-27B-4bit"
export XMLX_VLM_PORT="5118"
export XMLX_VLM_CHAT_PORT="5119"
ENVEOF
        ok "Environment variables saved to $SHELL_PROFILE"
    fi
fi

# ─── 10. Pre-download default model (optional) ──────────────────────────────
read -r -p "Pre-download default model (~20GB)? This avoids wait on first start. [y/N] " response </dev/tty || true
if [[ "$response" =~ ^[Yy]$ ]]; then
    info "Downloading default model (mlx-community/Qwen3.8-27B-4bit)..."
    info "This will take 10-30 minutes depending on your connection."
    .venv/bin/python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='mlx-community/Qwen3.8-27B-4bit', local_dir='./models/Qwen3.8-27B-4bit', local_dir_use_symlinks=False)
" || warn "Model download failed (will retry on first server start)"
    ok "Model download complete"
fi

# ─── 11. Done ───────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║           XMLX-VLM Installation Complete                            ║${NC}"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Repository:     ${NC}$SCRIPT_DIR"
echo -e "${GREEN}║  Virtual Env:    ${NC}$SCRIPT_DIR/.venv"
echo -e "${GREEN}║  Model API Port: ${NC}http://localhost:5118"
echo -e "${GREEN}║  Trading OS Web: ${NC}http://localhost:5119"
echo -e "${GREEN}║  API Key / Pwd:  ${NC}x123456"
echo -e "${GREEN}╠══════════════════════════════════════════════════════════════════════╣${NC}"
echo -e "${GREEN}║  Quick Start:                                                        ║${NC}"
echo -e "${GREEN}║    cd ${NC}$SCRIPT_DIR"
echo -e "${GREEN}║    ./service.sh start          # 启动模型推理与 Trading OS 终端      ║${NC}"
echo -e "${GREEN}║    ./service.sh status         # 查看服务运行状态                    ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════════════╝${NC}"
echo ""

read -r -p "Start the server now? [Y/n] " start_now </dev/tty || true
if [[ ! "$start_now" =~ ^[Nn]$ ]]; then
    info "Starting XMLX-VLM server..."
    cd "$SCRIPT_DIR"
    ./service.sh start
else
    info "You can start later with: cd $SCRIPT_DIR && ./service.sh start"
fi
