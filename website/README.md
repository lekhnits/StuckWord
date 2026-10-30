# Project page

The page for "Stuck on the Word", my PiEEG Octopus 16 Challenge project. Plain HTML, CSS and
JavaScript, no build step.

## Publish on GitHub Pages

1. Push the repository to GitHub.
2. On GitHub: Settings → Pages → Source: **GitHub Actions**.
3. The workflow in `.github/workflows/pages.yml` publishes this folder on every push to
   `main`. The page appears at `https://<github-user>.github.io/<repository>/`.

## Look at it locally

Open `index.html` in a browser, or serve the folder:

```bash
python -m http.server 8000 --directory website
```

## Add an update

New entries go at the top of the list in the `Updates` section of `index.html`.
