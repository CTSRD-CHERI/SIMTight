SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
SIMTIGHT=${SCRIPT_DIR}/..
CHERI_PATH="$SIMTIGHT/cheri-tools/cheri/output/sdk/bin"; 
export PATH=$CHERI_PATH:$PATH
echo $PATH
