# NeoVim Mojo Setup Guide

## What We Just Fixed

### 1. Added Filetype Plugin (`~/.config/nvim/after/ftplugin/mojo.lua`)
- 4-space indentation
- Smart indent for code blocks
- Python-like indent rules (Mojo syntax is similar)

### 2. Enabled Tree-sitter Indent (`~/.config/nvim/lua/plugins/mojo.lua`)
- Uses Python's tree-sitter grammar
- Auto-indent based on syntax

---

## Testing the Setup

### Step 1: Restart NeoVim
```bash
# Close all nvim instances and restart
cd /home/maskkiller/dev-coffee/repos/visage-ml
nvim src/block0/01_vector_matrix_ops/02_dot_product.mojo
```

### Step 2: Check LSP is Running
```vim
" Inside nvim, run:
:LspInfo

" You should see:
" Client: mojo (id 1)
" - filetypes: mojo
" - cmd: /path/to/.pixi/envs/default/bin/mojo-lsp-server -I .
" - root_dir: /home/maskkiller/dev-coffee/repos/visage-ml
```

### Step 3: Test Auto-Indent
Try typing this in a new Mojo file:

```mojo
fn test() raises:
    var x = 10  <Enter>
    |  ← Cursor should be indented here automatically!
```

### Step 4: Check Indent Settings
```vim
" Inside a .mojo file:
:set shiftwidth?    " Should show: shiftwidth=4
:set expandtab?     " Should show: expandtab
:set smartindent?   " Should show: smartindent
```

---

## LSP Capabilities to Verify

In a Mojo file, try:
- `gd` - Go to definition (should jump to function)
- `K` - Hover documentation (shows function signature)
- `<leader>ca` - Code actions (if available)
- `<leader>rn` - Rename symbol

---

## If Auto-Indent Still Doesn't Work

### Option A: Use vim-sleuth
```lua
-- Add to your plugins
{ "tpope/vim-sleuth" }  -- Auto-detects indent settings
```

### Option B: Manual Python Indent File
```bash
# Use Python's indent file for Mojo
mkdir -p ~/.config/nvim/after/indent
ln -s ~/.config/nvim/after/indent/python.vim ~/.config/nvim/after/indent/mojo.vim
```

### Option C: Check if cindent works better
```lua
-- In ~/.config/nvim/after/ftplugin/mojo.lua, try:
vim.bo.cindent = true        -- C-style indent
vim.bo.cinkeys = "0{,0},0),0],:,0#,!^F,o,O,e"
vim.bo.cinoptions = "(0,u0,U0"
```

---

## Debugging LSP

### Check if LSP starts on Mojo files:
```vim
:lua =vim.lsp.get_active_clients()
```

### View LSP logs:
```vim
:lua vim.cmd('e ' .. vim.lsp.get_log_path())
```

### Restart LSP manually:
```vim
:LspRestart mojo
```

---

## Expected Behavior After Fix

When you hit `<Enter>` after a line with `:`, the next line should:
1. Automatically indent to match the block
2. Maintain indentation for same-level statements
3. Dedent after return/closing braces

**Example:**
```mojo
fn my_function() raises:
    var x = 10  <Enter>
    |  ← Auto-indented here!
    if x > 5:  <Enter>
        |  ← Auto-indented one more level!
```

---

## Quick Commands Reference

```vim
" Check filetype
:set filetype?

" Check indent settings
:verbose set indentexpr?
:verbose set cindent?

" Reload config
:source ~/.config/nvim/lua/plugins/mojo.lua

" Format current file (manual)
:lua vim.lsp.buf.format()
```
