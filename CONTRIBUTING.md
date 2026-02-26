# Contributing

Simple rules to keep the repo clean. Please read before pushing anything.

---

## Branches

Never push directly to `main`. Always create a new branch for your work.

```bash
# Create your branch (use your member number and what you're working on)
git checkout -b member2/pressure-solver
git checkout -b member4/3d-cnn-model
git checkout -b member5/mode-toggle-ui

# Push your branch
git push origin your-branch-name
```

When your work is ready, open a **Pull Request** into `main` and tag at least one other member to review it.

**Branch naming:** `member<number>/<short-description>`
Examples: `member3/data-pipeline`, `member6/benchmark-script`

---

## Before You Push

Run this and make sure it doesn't crash:

```bash
python main.py --mode headless --frames 20
```

If that breaks after your changes, fix it before pushing.

---

## Code Rules

**No loops over grid cells.** Use NumPy slicing instead.
```python
# Bad
for x in range(32):
    for y in range(32):
        grid[x, y] = ...

# Good
grid[1:-1, 1:-1] = ...
```

**No hardcoded 32s.** Use `self.N` or `grid.N`.
```python
# Bad
pressure = np.zeros((32, 32, 32))

# Good
pressure = np.zeros((self.N, self.N, self.N))
```

**Don't modify files you don't own.** Each member has their files listed in `GUIDE.md`. If you need a change in someone else's file, talk to them first.

**Delete debug prints before pushing.**
```python
# Don't leave these in
print("here")
print(array.shape)
```

---

## Commit Messages

Keep them short and clear. Start with what you did.

```bash
# Good
git commit -m "add conjugate gradient solver option"
git commit -m "fix boundary condition on z-faces"
git commit -m "export model to onnx format"

# Bad
git commit -m "fix"
git commit -m "changes"
git commit -m "asdfgh"
```

---

## Don't Commit These

The `.gitignore` should already cover most of this, but just in case:

- `data/` — the `.npy` training files are too large for GitHub
- `.venv/` — everyone creates their own virtual environment
- `__pycache__/` — Python cache folders
- `*.onnx` — share model files separately (Google Drive, etc.)
- `*.pyc` — compiled Python files

---

That's it. Branch → code → test → pull request.
