#!/bin/bash

echo "Checking required tools..."

# Load requirements from a YAML file
REQUIREMENTS_FILE="requirements.yml"

if [ ! -f "$REQUIREMENTS_FILE" ]; then
  echo "Requirements file '$REQUIREMENTS_FILE' not found!" >&2
  exit 1
fi

# Parse the YAML file to extract the list of requirements
REQUIREMENTS=$(grep '^ *-' "$REQUIREMENTS_FILE" | sed -E 's/^ *- *(\S+).*$/\1/')

MISSING=()
for TOOL in $REQUIREMENTS; do
  if ! command -v "$TOOL" > /dev/null; then
    MISSING+=("$TOOL")
    echo "checking $TOOL. Missing."
  else
    echo "checking $TOOL. OK."
  fi
done

if [ ${#MISSING[@]} -ne 0 ]; then
  echo "The following tools are missing:" >&2
  for TOOL in "${MISSING[@]}"; do
    echo "  - $TOOL" >&2
  done
  echo "Please install the missing dependencies." >&2
  exit 1
else
  echo "All required tools are installed."
fi
