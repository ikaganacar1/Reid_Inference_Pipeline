#!/bin/bash
# Start npm with clean environment, preserving only necessary variables

# Preserve essential variables
PRESERVED_PATH="$PATH"
PRESERVED_HOME="$HOME"
PRESERVED_USER="$USER"
PRESERVED_SHELL="$SHELL"
PRESERVED_TERM="$TERM"

# Start with clean env and only add what we need
env -i \
  PATH="$PRESERVED_PATH" \
  HOME="$PRESERVED_HOME" \
  USER="$PRESERVED_USER" \
  SHELL="$PRESERVED_SHELL" \
  TERM="$PRESERVED_TERM" \
  HOST=localhost \
  PORT=8009 \
  REACT_APP_API_URL="http://localhost:8000" \
  npm start
