# Contributing to PicturaAI

Thanks for your interest in improving PicturaAI! Contributions of all kinds are welcome — bug reports, feature ideas, documentation, and code.

## Getting Started

1. Fork the repository and clone your fork.
2. Set up the local environment:
   ```bash
   python -m venv venv
   venv\Scripts\pip install -r backend\requirements.txt   # Windows
   # or: venv/bin/pip install -r backend/requirements.txt  # macOS/Linux
   ```
3. Run the app:
   ```bash
   python run.py
   ```
4. Open http://localhost:8000/app

## Making Changes

- Create a branch: `git checkout -b feature/your-feature`
- Keep changes focused and small where possible.
- Match the existing code style (the backend uses type hints and clear section comments).
- Test your changes locally before opening a PR — run a real style transfer to confirm the pipeline still works.

## Memory Constraints

PicturaAI is tuned to run on Render's 512 MB free tier. If you change the image pipeline, be mindful of peak memory:

- Inference resolution is controlled by `NST_MAX_DIM` (default `320`).
- Avoid holding multiple large tensors in memory simultaneously.
- Call `gc.collect()` after large operations when appropriate.

## Submitting a Pull Request

1. Push your branch to your fork.
2. Open a PR against `main` with a clear description of what changed and why.
3. Reference any related issues.

## Reporting Bugs

Open an issue with:
- What you expected to happen
- What actually happened
- Steps to reproduce
- Screenshots or logs if relevant

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
