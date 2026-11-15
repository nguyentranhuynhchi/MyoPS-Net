#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường tự động hóa cho MyoPS-Net trên Google Colab CLI
# Repository: https://github.com/nguyentranhuynhchi/MyoPS-Net.git
# Tập dữ liệu: MyoPS380 (304 Train / 76 Test, 300 Epochs, No-Val) & MyoPS++ (64 Train / 16 Test)
# Ghi chú: STRICTLY NO train command inside this script!
# ==============================================================================

set -e

echo "=== [1/6] Kiểm tra và Kích hoạt GPU NVIDIA ==="
cd /content

if [ -d "/usr/lib64-nvidia" ]; then
    if ! grep -q "/usr/lib64-nvidia" /etc/ld.so.conf.d/nvidia.conf 2>/dev/null; then
        echo "/usr/lib64-nvidia" >> /etc/ld.so.conf.d/nvidia.conf
        ldconfig 2>/dev/null || true
    fi
    export LD_LIBRARY_PATH=/usr/lib64-nvidia:${LD_LIBRARY_PATH}
fi
if ! grep -q "/usr/lib64-nvidia" /root/.bashrc 2>/dev/null; then
    echo 'export LD_LIBRARY_PATH=/usr/lib64-nvidia:$LD_LIBRARY_PATH' >> /root/.bashrc
fi

if ! command -v nvidia-smi &> /dev/null; then
    echo "⚠️ CẢNH BÁO: Chưa tìm thấy NVIDIA GPU. Hãy kiểm tra lại Runtime Type trên Colab!"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
fi

echo "=== [2/6] Mount Google Drive (Lưu Checkpoint an toàn) ==="
python3 -c "
import os
if not os.path.exists('/content/drive/MyDrive'):
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print('✅ Mount Google Drive thành công!')
    except Exception as e:
        print('⚠️ Không thể mount Drive tự động (bỏ qua nếu không dùng):', e)
" || true

echo "=== [3/6] Điều hướng thư mục MyoPS-Net ==="
if [ ! -d "/content/MyoPS-Net" ]; then
    echo "Thư mục /content/MyoPS-Net chưa tồn tại. Clone repo..."
    git clone https://github.com/nguyentranhuynhchi/MyoPS-Net.git /content/MyoPS-Net
else
    cd /content/MyoPS-Net
    CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "main")
    echo "Thư mục /content/MyoPS-Net đã tồn tại (nhánh hiện tại: $CURRENT_BRANCH). Tiến hành cập nhật..."
    if [ "$CURRENT_BRANCH" != "HEAD" ]; then
        git pull origin "$CURRENT_BRANCH" || true
    fi
fi
cd /content/MyoPS-Net

echo "=== [4/6] Cài đặt Dependencies cho MyoPS-Net ==="
python3 -m pip install --upgrade pip -q
if [ -f "/content/MyoPS-Net/requirements.txt" ]; then
    python3 -m pip install -r /content/MyoPS-Net/requirements.txt gdown -q
else
    python3 -m pip install torch torchvision numpy scipy h5py pandas nibabel medpy SimpleITK tqdm gdown -q
fi

echo "=== [5/6] Tải Dataset MyoPS380 từ Google Drive ==="
DATASET_FOLDER_ID="13T6892xy-pUwC1BSjGl4sRxoypMu-SIS"
DOWNLOAD_DIR="/content/dataset_download"
TARGET_DATASET_DIR="/content/MyoPS380_dataset"

mkdir -p "$DOWNLOAD_DIR"
mkdir -p "$TARGET_DATASET_DIR"

# Kiểm tra nếu dataset đã có sẵn trên Google Drive hoặc các repo khác để copy trực tiếp
for DRIVE_CANDIDATE in \
    "/content/drive/MyDrive/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/Processed_data" \
    "/content/drive/MyDrive/NCKH_V2/MyoPS380_dataset/Processed_data" \
    "/content/drive/MyDrive/NCKH/MyoPS380_dataset/Processed_data" \
    "/content/SCAR/MyoPS380_dataset/Processed_data" \
    "/content/IMS2Trans/MyoPS380_dataset/Processed_data"; do
    if [ -d "$DRIVE_CANDIDATE" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        echo "✅ Phát hiện dataset tại: $DRIVE_CANDIDATE. Đang sao chép vào SSD cục bộ..."
        mkdir -p "$TARGET_DATASET_DIR/Processed_data"
        cp -r "$DRIVE_CANDIDATE/." "$TARGET_DATASET_DIR/Processed_data/"
        break
    fi
done

if [ -z "$(ls -A "$DOWNLOAD_DIR" 2>/dev/null)" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
    echo "Đang tải toàn bộ thư mục dataset từ Google Drive (Folder ID: $DATASET_FOLDER_ID)..."
    gdown --folder --id "$DATASET_FOLDER_ID" -O "$DOWNLOAD_DIR" --remaining-ok || {
        echo "⚠️ Tải folder qua gdown gặp cảnh báo, tiếp tục kiểm tra các tệp đã tải..."
    }
fi

if [ ! -d "$TARGET_DATASET_DIR/Processed_data/bSSFP" ]; then
    for zipfile in $(find "$DOWNLOAD_DIR" -maxdepth 2 -name "*.zip" 2>/dev/null); do
        echo "Đang giải nén $zipfile vào $TARGET_DATASET_DIR ..."
        command -v unzip &>/dev/null || (apt-get update -qq && apt-get install -y unzip)
        unzip -q -o "$zipfile" -d "$TARGET_DATASET_DIR/"
    done

    if [ -d "$DOWNLOAD_DIR/Processed_data" ] && [ -z "$(ls -A "$TARGET_DATASET_DIR/Processed_data" 2>/dev/null)" ]; then
        mkdir -p "$TARGET_DATASET_DIR/Processed_data"
        cp -r "$DOWNLOAD_DIR/Processed_data/." "$TARGET_DATASET_DIR/Processed_data/"
    fi
fi

FOUND_TRAIN_NPZ=$(find "$TARGET_DATASET_DIR" "$DOWNLOAD_DIR" -type d -name "train_npz" 2>/dev/null | grep -i "bSSFP" | head -n 1 || true)
if [ -n "$FOUND_TRAIN_NPZ" ]; then
    LOCAL_DATASET=$(dirname "$(dirname "$FOUND_TRAIN_NPZ")")
    echo "✅ Đã xác định thư mục dataset chuẩn (Processed_data) tại: $LOCAL_DATASET"
else
    LOCAL_DATASET="/content/MyoPS380_dataset/Processed_data"
    echo "⚠️ Chưa phát hiện thư mục bSSFP/train_npz, đặt mặc định: $LOCAL_DATASET"
fi

export MYOPS380_DATA_ROOT="$LOCAL_DATASET"
sed -i '/^export MYOPS380_DATA_ROOT=/d' /root/.bashrc 2>/dev/null || true
echo "export MYOPS380_DATA_ROOT=\"$LOCAL_DATASET\"" >> /root/.bashrc

echo "=== [6/6] Cài đặt tmux và Kiểm tra PyTorch ==="
command -v tmux &> /dev/null || (apt-get update -qq && apt-get install -y tmux)
python3 -c "import torch; print(f'PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo "=== HOÀN TẤT THIẾT LẬP MÔI TRƯỜNG CHO MYOPS-NET ==="
echo ""
echo "💡 HƯỚNG DẪN CHẠY ĐO HIỆU SUẤT TÍNH TOÁN (1 SEED ĐỘC LẬP 1234 TRÊN 1 GPU A100):"
echo "  1. Mở session tmux mới:"
echo "     tmux new -s myopsnet_eff"
echo ""
echo "  2. Chạy đo hiệu suất độc lập (FP32 thuần, num_workers=0, pin_memory=True):"
echo "     cd /content/MyoPS-Net"
echo "     python3 benchmark_efficiency.py --dataset myops380 --amp none --num_workers 0 --pin_memory --output outputs/runs/myopsnet/efficiency_metrics.json"
echo ""
echo "  3. Hoặc chạy huấn luyện chuẩn 300 epochs và đo hiệu suất:"
echo "     python3 train.py --dataset myops380 --data_root \"$LOCAL_DATASET\" --split_dir preprocessing/splits/myops380_no_val --run_dir runs/myopsnet_seed1234 --seed 1234 --batch_size 16 --epochs 300 --amp none --num_workers 0 --pin_memory --measure_efficiency"
echo ""
echo "  4. Đánh giá kiểm thử (predict & evaluate):"
echo "     python3 predict.py --dataset myops380 --checkpoint runs/myopsnet_seed1234/checkpoints/myopsnet_seed1234.pt --data_root \"$LOCAL_DATASET\" --run_dir runs/myopsnet_seed1234"
echo "     python3 evaluate_predictions.py --dataset myops380 --pred_dir runs/myopsnet_seed1234/test_predictions --data_root \"$LOCAL_DATASET\" --output_dir runs/myopsnet_seed1234/evaluation"
echo ""
echo "  5. Tách khỏi tmux (detach): Nhấn Ctrl+B rồi thả ra, sau đó nhấn D"
echo "  6. Xem lại tiến trình: tmux attach -t myopsnet_eff"
echo ""
