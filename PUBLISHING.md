# Publish the repository bundle

The bundle contains the repository files, including GitHub Actions and Git
configuration files, without a virtual environment, caches, or Git history.
Extract it and open a terminal in `CVI_IJOC/`.

To publish to the currently empty repository after authenticating with a GitHub
account that has write access:

```bash
git init -b main
git add .
git commit -m "Add reproducible PFAOS and MC-PFAOS experiments"
git remote add origin https://github.com/qluorocky/CVI_IJOC.git
git push -u origin main
```

If you are already working in a clone, copy the extracted files into that
clone, inspect `git status`, and commit and push there. Preserve any changes
that have subsequently been made on GitHub; do not force-push over them.

For publication through ChatGPT, the GitHub connection must include this
repository under the `qluorocky` account and allow repository-content writes.
