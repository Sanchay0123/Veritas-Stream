#!/bin/bash

# --- CONFIGURATION ---
WINDOWS_DRIVE_LETTER="E:"   # How Windows sees it
LINUX_MOUNT_POINT="/mnt/e"  # Where we want it in Linux
VAULT_FOLDER_NAME="veritas_vault"
PROJECT_VAULT="./storage/vault"
# ---------------------

EXTERNAL_PATH="$LINUX_MOUNT_POINT/$VAULT_FOLDER_NAME"

echo "🔍 Checking for External Drive ($WINDOWS_DRIVE_LETTER)..."

# 1. Check if WSL can see the drive content. If not, force mount it.
if [ -z "$(ls -A $LINUX_MOUNT_POINT 2>/dev/null)" ]; then
    echo "⚠️  Drive not visible in Linux yet. Attempting to mount..."
    
    # Ensure the directory exists
    if [ ! -d "$LINUX_MOUNT_POINT" ]; then
        sudo mkdir -p "$LINUX_MOUNT_POINT"
    fi

    # The Magic Command: Connect Windows Drive to Linux Folder
    sudo mount -t drvfs "$WINDOWS_DRIVE_LETTER" "$LINUX_MOUNT_POINT"
    
    if [ $? -ne 0 ]; then
        echo "❌ Error: Could not mount drive. Is it plugged in?"
        exit 1
    fi
    echo "✅ Drive mounted to Linux successfully."
else
    echo "✅ Drive is already visible to Linux."
fi

# 2. Check/Create the vault folder on the disk
if [ ! -d "$EXTERNAL_PATH" ]; then
    echo "   Creating '$VAULT_FOLDER_NAME' folder on disk..."
    mkdir -p "$EXTERNAL_PATH"
fi

# 3. Bind the project folder to the disk
if mountpoint -q "$PROJECT_VAULT"; then
    echo "ℹ️  Vault is ALREADY connected."
else
    echo "🔗 Connecting Project Vault to External Disk..."
    sudo mount --bind "$EXTERNAL_PATH" "$PROJECT_VAULT"
    
    if [ $? -eq 0 ]; then
        echo "✅ SUCCESS: Vault is online!"
        echo "📂 Storage location: $EXTERNAL_PATH"
    else
        echo "❌ FAILED to bind vault folder."
    fi
fi
