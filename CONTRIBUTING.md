# Environment Setup

## Version of Tools

| Item    | Version |
| ------- | ------- |
| typst   | 0.14.2  |
| node.js | 24.13.1 |
| python  | 3.11.2  |

<!-- 到時候開發的時候要先確定要使用的 versions ，然後先在 setup 分支更新 Docs-->

## Requirements

Please install `pre-commit` and formatters before making any commit.

1. Python

```sh
uv sync
```

2. Node.js

```sh
npm install
```

3. Pre-commit

```sh
uv run pre-commit install
```

```sh
uv run pre-commit clean
uv run pre-commit autoupdate
uv run pre-commit run --all-files
```

## Formatters

> [!TIP]
> We highly recommend enabling auto-format on save in your editor (e.g. Neovim with `conform.nvim`).

1. Python: Using `Isort` and `Black`.
2. React: Using `Prettier`.
3. Markdown: Using `Prettier`.
4. Typst: Using `Typstyle`.

# Commit Workflow

To maintain code quality, we strictly enforce formatting via `pre-commit`.

> [!IMPORTANT]
> Ensure you have executed `pre-commit install` locally.

When you run `git commit`, the system will automatically check your changed files:

- If the check fails, Review the error message. In most cases, the tools will fix the formatting for you. Simply `git add` the upadted files and commit again.
- Do not use `--no-verify`. Commits that bypass local hooks will be rejected by the GitHub Actions CI process.

# Pull Request Rules

- Please complete the checklist in `pull_request_template.md` when opening a PR. It'll ask you when you opening any PR.
- Ensure all CI checks are passing (indicated by a green tick) before requesting a review.

# Troubleshooting

## Windows Encoding Issue (`UnicodeDecodeError`)

If you are using Windows and encounter a `UnicodeDecodeError: 'cp950' codec can't decode...` when running `pre-commit` or `pre-commit autoupdate`, it is due to a mismatch between the Windows system locale (CP950) and the UTF-8 encoding used by Python/Git.

### Solution: Enable Python UTF-8 Mode

You can fix this by forcing Python to use UTF-8 mode.

#### **Temporary Fix (Current Session)**

In your terminal (PowerShell), run:

```powershell
$env:PYTHONUTF8 = 1
uv run pre-commit run --all-files
```

### Permanent Fix (Recommended)

To avoid setting this every time, add it to your System Environment Variables:

1. Open the Start Menu, search for "Edit the system environment variables", and press Enter.
2. Click the Environment Variables button.
3. Under User variables, click New.
4. Variable name: PYTHONUTF8
5. Variable value: 1
6. Click `OK` and restart your terminal or Neovim.

> [!TIP]
> Alternatively, you can run chcp 65001 in your terminal to switch the active code page to UTF-8 before running git commands.
