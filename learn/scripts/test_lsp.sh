#!/bin/bash
# Test if Mojo LSP server works

echo "Testing Mojo LSP Server..."
echo ""

# Check if LSP exists
if [ -f ".pixi/envs/default/bin/mojo-lsp-server" ]; then
    echo "✓ LSP server found: .pixi/envs/default/bin/mojo-lsp-server"
else
    echo "✗ LSP server NOT found!"
    exit 1
fi

# Check if it's executable
if [ -x ".pixi/envs/default/bin/mojo-lsp-server" ]; then
    echo "✓ LSP server is executable"
else
    echo "✗ LSP server is not executable!"
    exit 1
fi

# Try to get version/help
echo ""
echo "LSP Server Info:"
.pixi/envs/default/bin/mojo-lsp-server --version 2>&1 || echo "(No version flag)"
.pixi/envs/default/bin/mojo-lsp-server --help 2>&1 | head -10 || echo "(No help available)"

echo ""
echo "✅ LSP server is ready!"
echo ""
echo "Next: Open a .mojo file in nvim and run :LspInfo to verify it's attached"
