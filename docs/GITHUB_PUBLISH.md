# Publish this repository on GitHub

This working folder is initialized as a Git repository and ready to push. Real datasets are ignored by `.gitignore`.

## GitHub Desktop (easiest on Windows)

1. Extract the downloaded ZIP to a folder on your PC.
2. Open GitHub Desktop → **File → Add local repository** and select the extracted folder.
3. If GitHub Desktop reports there is no repository, choose **Create a Repository** for that folder.
4. Choose **Publish repository**. Recommended visibility: **Private** until coursework submission and data usage policies are reviewed.
5. Copy the repository URL and share it with your teammate. Add the teammate under **Settings → Collaborators** in the GitHub repository.

## Using command line

If you create a new blank repository at GitHub first (do not check README or .gitignore since they already exist here):

```bash
git init -b main
git add .
git commit -m "Initial content-based recommender and evaluation"
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

If the extracted folder already has a `.git` directory, skip `git init` and the initial commit. Never place access tokens in code or shell history.

## What the teammate changes

Add `boardrec/memory_cf.py`, `boardrec/svd.py`, `boardrec/hybrid.py` and corresponding tests. Use the existing CSV split and the `predict(user_id, game_id)` contract. See `docs/TEAM_HANDOFF.md`.
