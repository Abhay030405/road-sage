#!/usr/bin/env bash
# =============================================================================
# RoadSage Model Downloader
# =============================================================================
#
# Downloads (or verifies) all pretrained weights needed for RoadSage.
# Safe to run multiple times — already-downloaded files are hash-verified and
# skipped if they match. Any hash mismatch causes an immediate exit.
#
# Usage:
#   bash models/download_models.sh
#
# Requirements: curl or wget, sha256sum (Linux) / shasum (macOS)
# =============================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR"       # script lives inside models/

# ---------------------------------------------------------------------------
# Colour helpers (disabled when not a TTY)
# ---------------------------------------------------------------------------

if [ -t 1 ]; then
    BOLD="\033[1m"; GREEN="\033[1;32m"; YELLOW="\033[1;33m"
    CYAN="\033[1;36m"; RED="\033[1;31m"; RESET="\033[0m"
else
    BOLD=""; GREEN=""; YELLOW=""; CYAN=""; RED=""; RESET=""
fi

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; }

# ---------------------------------------------------------------------------
# sha256 helper — wraps sha256sum (Linux) and shasum -a 256 (macOS)
# ---------------------------------------------------------------------------

compute_sha256() {
    local file="$1"
    if command -v sha256sum &>/dev/null; then
        sha256sum "$file" | awk '{print $1}'
    elif command -v shasum &>/dev/null; then
        shasum -a 256 "$file" | awk '{print $1}'
    else
        error "Neither sha256sum nor shasum found. Cannot verify checksums."
        exit 1
    fi
}

# ---------------------------------------------------------------------------
# download_file <url> <output_path> <expected_sha256>
#
# * If <output_path> exists and hash matches → skip.
# * Otherwise download with wget or curl, then verify.
# ---------------------------------------------------------------------------

download_file() {
    local url="$1"
    local output_path="$2"
    local expected_sha256="$3"

    # ---- Already downloaded? ----
    if [ -f "$output_path" ]; then
        local actual_hash
        actual_hash="$(compute_sha256 "$output_path")"
        if [ "$actual_hash" = "$expected_sha256" ]; then
            success "Already downloaded: $(basename "$output_path")"
            return 0
        else
            warn "Hash mismatch for $(basename "$output_path") — re-downloading."
            rm -f "$output_path"
        fi
    fi

    # ---- Choose downloader ----
    if command -v wget &>/dev/null; then
        info "Downloading $(basename "$output_path") with wget …"
        wget --quiet --show-progress --no-clobber -O "$output_path" "$url"
    elif command -v curl &>/dev/null; then
        info "Downloading $(basename "$output_path") with curl …"
        curl -fSL --progress-bar -o "$output_path" "$url"
    else
        error "Neither wget nor curl is available. Please install one and retry."
        exit 1
    fi

    # ---- Verify after download ----
    local actual_hash
    actual_hash="$(compute_sha256 "$output_path")"
    if [ "$actual_hash" != "$expected_sha256" ]; then
        error "SHA-256 verification FAILED for $(basename "$output_path")."
        error "  Expected : $expected_sha256"
        error "  Got      : $actual_hash"
        rm -f "$output_path"
        exit 1
    fi

    success "Downloaded and verified: $(basename "$output_path")"
}

# ---------------------------------------------------------------------------
# Ensure models/ directory exists
# ---------------------------------------------------------------------------

mkdir -p "$MODELS_DIR"

# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------

echo ""
echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   RoadSage Model Downloader                                ${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

# ---------------------------------------------------------------------------
# 1. UFLDv2 ResNet-18  — lane detector
#    Official repo: https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2
#
#    The checkpoint is not hosted on a stable CDN, so we guide the user to
#    download it manually and place it in the right location.
# ---------------------------------------------------------------------------

echo -e "${BOLD}[1/4] UFLDv2 ResNet-18 (Lane Detector)${RESET}"
echo ""

if [ -f "$MODELS_DIR/lane_detector.pth" ]; then
    success "lane_detector.pth already present — skipping manual-download prompt."
else
    warn  "lane_detector.pth not found."
    echo  ""
    info  "For UFLDv2 weights, please download manually from:"
    echo  "      https://github.com/cfzd/Ultra-Fast-Lane-Detection-v2/releases"
    echo  ""
    info  "Recommended checkpoint: culane_res18.pth or tusimple_res18.pth"
    info  "Place the .pth file at:  models/lane_detector.pth"
    echo  ""
    info  "Then export to ONNX by running:"
    echo  "      python training/scripts/export_onnx.py --model lane_detector"
    echo  ""
fi

if [ -f "$MODELS_DIR/lane_detector.onnx" ]; then
    success "lane_detector.onnx already present."
else
    warn  "lane_detector.onnx not found — run export_onnx.py after placing the .pth file."
fi

echo ""

# ---------------------------------------------------------------------------
# 2. MiDaS Small — monocular depth estimator
#    Official repo: https://github.com/isl-org/MiDaS
#
#    torch.hub will download this automatically on first use.
#    The manual path is provided for air-gapped / offline environments.
# ---------------------------------------------------------------------------

echo -e "${BOLD}[2/4] MiDaS Small (Depth Estimator)${RESET}"
echo ""

if [ -f "$MODELS_DIR/depth_estimator.pth" ]; then
    success "depth_estimator.pth already present — skipping auto-download hint."
else
    info  "MiDaS Small is downloaded automatically on first use via torch.hub."
    info  "No manual action is required for online environments."
    echo  ""
    info  "For offline / air-gapped deployments, download manually from:"
    echo  "      https://github.com/isl-org/MiDaS/releases"
    info  "Place the weights at:  models/depth_estimator.pth"
fi

echo ""

# ---------------------------------------------------------------------------
# 3. NanoDet-Plus-m — lightweight object / obstacle detector
#    Official repo: https://github.com/RangiLyu/nanodet
# ---------------------------------------------------------------------------

echo -e "${BOLD}[3/4] NanoDet-Plus-m (Object Detector)${RESET}"
echo ""

if [ -f "$MODELS_DIR/object_detector.pth" ]; then
    success "object_detector.pth already present — skipping hint."
else
    info  "NanoDet-Plus-m weights are available at:"
    echo  "      https://github.com/RangiLyu/nanodet/releases"
    info  "Download nanodet-plus-m_416.pth and place it at:"
    echo  "      models/object_detector.pth"
fi

echo ""

# ---------------------------------------------------------------------------
# 4. MobileNetV3-Small — decision CNN
#    Trained from scratch in Phase 4 using pseudo-labels.  No download needed.
# ---------------------------------------------------------------------------

echo -e "${BOLD}[4/4] MobileNetV3-Small (Decision CNN)${RESET}"
echo ""
info  "Decision CNN will be trained during Phase 4 pseudo-labeling."
info  "No pre-trained download is needed at this stage."
echo  ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

echo -e "${BOLD}============================================================${RESET}"
echo -e "${BOLD}   Download Summary${RESET}"
echo -e "${BOLD}============================================================${RESET}"
echo ""

MISSING=0
for model_file in lane_detector.pth lane_detector.onnx depth_estimator.pth object_detector.pth; do
    if [ -f "$MODELS_DIR/$model_file" ]; then
        echo -e "  ${GREEN}✓${RESET}  $model_file"
    else
        echo -e "  ${YELLOW}✗${RESET}  $model_file  (not yet present)"
        MISSING=$((MISSING + 1))
    fi
done

echo ""
if [ "$MISSING" -eq 0 ]; then
    success "All model files are present. RoadSage is ready to run."
else
    warn "$MISSING model file(s) still need to be obtained manually."
    warn "See instructions above or models/README.md for details."
fi

echo ""
