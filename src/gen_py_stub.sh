#!/bin/bash
# Helper script to generate a py stub that already contains the license information.

set -e

if [ $# -eq 0 ]
  then
    echo "USAGE

./gen_py_stub.sh FILENAME [AUTHOR_FULL AUTHOR_SHORT CONTACT VERSION]

This script can be used to generate a template phython file.
The FILENAME always has to be specified. AUTHOR, CONTACT and VERSION are
optional arguments."

    exit 0
fi

FILENAME=$1
if ! [[ "$1" == *\.py ]]
  then
    FILENAME="${FILENAME}.py"
fi

AUTHOR_FULL=$2
if [ -z "$2" ]
  then
    AUTHOR_FULL='Christian Henning'
    echo "No AUTHOR_FULL supplied. Using $AUTHOR_FULL instead."
fi

AUTHOR_SHORT=$3
if [ -z "$3" ]
  then
    AUTHOR_SHORT='ch'
    echo "No AUTHOR_SHORT supplied. Using $AUTHOR_SHORT instead."
fi

CONTACT=$4
if [ -z "$4" ]
  then
    CONTACT='christian@ini.ethz.ch'
    echo "No CONTACT supplied. Using $CONTACT instead."
fi

VERSION=$5
if [ -z "$5" ]
  then
    VERSION='1.0'
    echo "No VERSION supplied. Using $VERSION instead."
fi

PY_VERSION="$(python3 -c 'import platform; print(platform.python_version())')"

echo "#!/usr/bin/env python3
# Copyright $(date +"%Y") $AUTHOR_FULL
#
# Licensed under the Apache License, Version 2.0 (the \"License\");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an \"AS IS\" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
\"\"\"
@title           :$FILENAME
@author          :$AUTHOR_SHORT
@contact         :$CONTACT
@created         :$(date +"%m/%d/%Y")
@version         :$VERSION
@python_version  :$PY_VERSION

TODO: SUMMARY

TODO: DETAILED DESCRIPTION
\"\"\"

import configuration as config

import logging
logger = logging.getLogger(config.logging_name)

if __name__ == '__main__':
    pass

" > $FILENAME

echo "Stub generated as file $FILENAME"
