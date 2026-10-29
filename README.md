# NEL Player Lab

Fantasy football stats, next-game and rest-of-season projections, and defense matchups for the National Ego League.
`update_lab.py` downloads the latest nflverse data and builds one web page (`NEL_Player_Lab.html`). A GitHub workflow runs it on a schedule and publishes the result with GitHub Pages.

See **SETUP_GITHUB.md** for the one-time setup.

## Files
- `update_lab.py`: downloads data and builds the page
- `lab_template.html`: the page design
- `model.json`: the projection model, trained on 2021 to 2025
- `analog.json`: about 14,600 past player-games, used for the "odds for a stat line" feature
- `train_model.py`: refits `model.json` and `analog.json` (rarely needed)
- `rosters.json`: your league's rosters. Edit this when rosters change.
- `.github/workflows/update.yml`: the schedule (Tuesday, Tuesday evening, Friday evening)

## Run it on your own computer instead
`pip install pandas numpy` then `python update_lab.py`, then open `NEL_Player_Lab.html`.

## Settings (top of update_lab.py)
`SEASON`, `LAST_WEEK`, `PLAYOFF_WEEKS` (your fantasy playoff weeks), `MATCHUP_STRENGTH_DEFAULT`.
