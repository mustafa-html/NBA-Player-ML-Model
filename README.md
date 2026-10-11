# NBA Playstyle Similarity Engine

Find NBA players who play the same way. Every qualifying player-season since 2015–16 is reduced to 17 playstyle rates, grouped into archetypes with K-Means, and searched by cosine similarity.

Search a player and you get his archetype, his closest matches, a radar chart comparing him to his best match, and his position on a two-dimensional map of the league.

```
Luka Dončić (2026): Primary Creator — creates his own threes, high-usage scorer, primary playmaker
   91.2%  Donovan Mitchell (2026, Primary Creator)
   90.5%  Cade Cunningham (2026, Primary Creator)
   88.5%  James Harden (2026, Primary Creator)
```

## Run it

```bash
pip install -r requirements-dev.txt
python -m playstyle.cli train --k 10     # ~5 seconds
flask --app wsgi:app run --port 5000     # http://localhost:5000
```

Eleven seasons of data are bundled in `data/raw`, so nothing is downloaded on first run. If you skip the training step, the web app trains on startup.

In production: `gunicorn -w 2 -b 0.0.0.0:8000 wsgi:app`, or `make docker`, which trains during the image build so the container serves straight away.

## How it works

**Style, not quality.** Every feature is a rate or a share: percentage of shots at the rim, share of threes from the corner, assist rate, block rate, usage. A bench player and an All-Star who play the same way land in the same place. The full list is in `playstyle/config.py`, with plain-English descriptions used to explain each player.

**Standardized per season.** The league's three-point rate has climbed steeply, so a 2016 player scored against 2026 norms would look like someone who refuses to shoot. Each feature is standardized within its own season, so a player is described relative to the league he actually played in.

**K-Means, then named archetypes.** Cluster centroids are matched to hand-written prototype directions ("Rim-Running Big", "Spot-Up Shooter", "Floor General") using the Hungarian algorithm, so every cluster gets a distinct, readable name instead of "Cluster 7". A centroid that matches nothing well falls back to "Versatile Role Player" rather than being forced into a bad label.

**Cosine similarity for matching.** Two players match when their strengths and weaknesses point the same way, whatever their overall level of extremeness. This is deliberately not the same as cluster membership: a player near a cluster boundary can have his best matches in a neighbouring archetype, which is usually the correct answer.

**Choosing k.** `--k auto` picks the best silhouette score, which lands on 4: the guard/wing/big split. That is a statistically clean answer and a useless one, so the default is `--k 10`, which separates archetypes that are actually distinct in basketball terms. `reports/k_selection.png` shows the trade-off. This is a judgement call, not a tuned result, and the code makes it easy to disagree: `--k auto`, any fixed `k`, or `N_CLUSTERS` as an environment variable.

## Layout

```
playstyle/
  config.py      feature definitions, archetype prototypes, training settings
  data.py        loading, and collapsing traded players into one row per season
  features.py    per-season standardization and plain-English trait descriptions
  model.py       K-Means, k scoring, archetype naming, persistence
  similarity.py  accent-insensitive name lookup and cosine nearest neighbours
  reports.py     Matplotlib charts (Agg backend, headless-safe)
  pipeline.py    train() and the Engine used by the app and the CLI
  cli.py         fetch / train / similar
webapp/          Flask app, JSON API, templates and styles
tests/           20 tests
```

Training and inference are separate: `train()` writes a model, a table and a vector matrix to `artifacts/`; `Engine.load()` reads them. The web app holds one `Engine` in memory. Nothing recomputes per request, and the same `Engine` backs the CLI, the API and the pages.

## CLI

```bash
python -m playstyle.cli fetch                          # refresh the data from the source repo
python -m playstyle.cli train --k 10                   # or --k auto
python -m playstyle.cli similar "Rudy Gobert"          # matches within the same season
python -m playstyle.cli similar "Klay Thompson" --season 2016 --scope all
```

Names are matched without accents or punctuation, so `luka doncic` finds Luka Dončić, and a miss suggests close names.

## API

| Endpoint | Returns |
|---|---|
| `GET /api/health` | Status, number of player-seasons, season range, cluster count |
| `GET /api/players?season=2026` | Player names |
| `GET /api/archetypes?season=2026` | Each archetype with its size and example players |
| `GET /api/similar?player=Jalen+Brunson&n=10&scope=season` | Profile plus ranked matches |
| `POST /api/infer` | Archetype and matches for a stat line not in the dataset |
| `GET /charts/radar.png`, `GET /charts/map.png` | Rendered charts |

`/api/infer` takes any subset of the features; anything you leave out is filled with that season's median.

```bash
curl -X POST localhost:5000/api/infer -H 'Content-Type: application/json' \
  -d '{"stats": {"three": 0.72, "corner3": 0.45, "usg": 13.5, "astp": 6.0}, "season": 2026}'
```

## Configuration

Environment variables: `FIRST_SEASON`, `LAST_SEASON`, `MIN_MINUTES` (default 500), `MIN_GAMES` (default 20), `N_CLUSTERS`, `RAW_DIR`, `ARTIFACT_DIR`, `REPORT_DIR`.

## Tests

`pytest` runs 20 tests in about six seconds, with no network access. They cover the awkward parts rather than the obvious ones:

- traded players, where Basketball Reference emits one row per team plus a combined row, in two different formats (`TOT` historically, `2TM`/`3TM` from 2025)
- undefined rates, such as the share of a centre's threes that were assisted when he made none
- per-season standardization, checked by confirming each season's features have mean 0 and standard deviation 1
- archetype naming, including that names stay unique and weak matches fall back
- a first-run test that builds a model and charts in an empty directory, which caught a bug where startup training ignored the configured report directory

## Data

Basketball Reference statistics via the [sumitrodatta/bball-reference-datasets](https://github.com/sumitrodatta/bball-reference-datasets) mirror. Seasons are labelled by the year they end, so 2026 is 2025–26. Only NBA seasons are included. Players need 500 minutes and 20 games to qualify, which leaves about 3,970 player-seasons from 948 players.

Because the current season is still in progress, its rates move as games are played. Run `make fetch && make train` to update.
