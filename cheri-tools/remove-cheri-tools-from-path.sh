SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CHERI_PATH="${SCRIPT_DIR}/cheri/output/sdk/bin\:"; PATH=$(echo "$PATH" | sed -e "s|$CHERI_PATH||g")
export PATH=$PATH
echo $PATH
