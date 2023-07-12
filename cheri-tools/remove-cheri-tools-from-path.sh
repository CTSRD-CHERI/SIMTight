CHERI_PATH="/workspace/cheri-tools/cheri/output/sdk/bin\:"; PATH=$(echo "$PATH" | sed -e "s|$CHERI_PATH||g")
export PATH=$PATH
echo $PATH
