# Contributing to Qwodel

Thank you for your interest in contributing to Qwodel! We welcome contributions from the community to help make model quantization accessible to everyone.

## Getting Started

1.  **Fork the repository** on GitHub.
2.  **Clone your fork** locally:
    ```bash
    git clone https://github.com/yourusername/qwodel.git
    cd qwodel
    ```
3.  **Install development dependencies**:
    ```bash
    pip install -e .[dev]
    ```

## Development Workflow

1.  Create a new branch for your feature or fix:
    ```bash
    git checkout -b feature/my-awesome-feature
    ```
2.  Make your changes.
3.  Run tests to ensure everything is working:
    ```bash
    pytest tests/
    ```
4.  Commit your changes with clear, descriptive messages.

## Pull Request Process

1.  Push your branch to GitHub.
2.  Open a Pull Request (PR) against the `main` branch of `qwodel`.
3.  Describe your changes in detail in the PR description.
4.  Wait for review and address any feedback.

## Coding Standards

-   **Type Hinting**: Use Python type hints for all function arguments and return values.
-   **Formatting**: We use `ruff` or `black` for code formatting.
-   **Documentation**: Ensure all new functions and classes have docstrings.

## Reporting Issues

If you find a bug or have a feature request, please open an issue on the GitHub repository. Please provide as much detail as possible, including steps to reproduce the issue.

Thank you for contributing!
