#!/bin/sh
#!/bin/bash

# Stop on error
set -e
# Stop when undefined variable is ecountered
set -u
# Easier to debug errors
set -o pipefail

SCRIPT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

pushd $SCRIPT_DIR > /dev/null 2>&1
  python3 -m unittest discover --pattern=*_test.py
popd > /dev/null 2>&1
