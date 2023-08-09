SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
CHERI_PATH="${SCRIPT_DIR}/cheri/output/sdk/bin"; 
export PATH=$CHERI_PATH:$PATH
echo $PATH
