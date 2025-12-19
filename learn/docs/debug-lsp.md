# Debug LSP Not Attaching

Run these commands in nvim with a .mojo file open:

## 1. Check Filetype
```vim
:set filetype?
```
Expected: `filetype=mojo`

## 2. Check if config loaded
```vim
:lua print(vim.inspect(require("lspconfig.configs").mojo))
```

## 3. Check available servers
```vim
:lua print(vim.inspect(require("lspconfig").util.available_servers()))
```

## 4. Manually start LSP
```vim
:lua require("lspconfig").mojo.setup({ cmd = { vim.fn.getcwd() .. "/.pixi/envs/default/bin/mojo-lsp-server", "-I", "." } })
:edit
```

## 5. Check for errors
```vim
:messages
:checkhealth lsp
```
