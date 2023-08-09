SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMTIGHT=${SCRIPT_DIR}/..
CHERI_PATH="$SIMTIGHT/cheri-tools/cheri/output/sdk/bin\:"; PATH=$(echo "$PATH" | sed -e "s|$CHERI_PATH||g")
export PATH=$PATH
echo $PATH
