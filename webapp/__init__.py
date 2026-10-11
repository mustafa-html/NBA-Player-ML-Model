"""Flask web app and JSON API."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, Response

from playstyle import config, reports
from playstyle.pipeline import Engine, ensure_artifacts, records


def create_app(artifact_dir: Optional[Path] = None, report_dir: Optional[Path] = None) -> Flask:
    artifact_dir = Path(artifact_dir or config.ARTIFACT_DIR)
    report_dir = Path(report_dir or config.REPORT_DIR)
    ensure_artifacts(artifact_dir, report_dir)
    engine = Engine.load(artifact_dir)
    latest = engine.seasons[-1]

    app = Flask(__name__)
    app.config["ENGINE"] = engine
    all_names = sorted(engine.table["player"].unique())

    @app.context_processor
    def inject_names():
        return {"all_names": all_names}

    def season_arg() -> Optional[int]:
        v = request.args.get("season", "")
        return int(v) if v.isdigit() else None

    def lookup(name: str, season: Optional[int]):
        idx = engine.index.find(name, season) if name else None
        if idx is None:
            return None, engine.index.suggest(name) if name else []
        return idx, []

    def archetype_overview(season: int):
        t = engine.table[engine.table["season"] == season]
        out = []
        for a in engine.model.archetypes:
            members = t[t["cluster"] == a["cluster"]].sort_values("mp", ascending=False)
            out.append({**a, "count": len(members), "examples": members["player"].head(4).tolist()})
        return sorted(out, key=lambda x: -x["count"])

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def index():
        return render_template(
            "index.html", seasons=engine.seasons[::-1], latest=latest,
            archetypes=archetype_overview(latest),
        )

    @app.get("/player")
    def player_page():
        name = request.args.get("name", "").strip()
        season = season_arg()
        scope = request.args.get("scope", "season")
        idx, suggestions = lookup(name, season)
        if idx is None:
            return render_template("not_found.html", name=name, season=season, suggestions=suggestions), 404
        profile = engine.player_profile(idx)
        matches = engine.similar(idx, n=10, scope=scope)
        others = engine.table[engine.table["player_id"] == profile["player_id"]]["season"].tolist()
        return render_template(
            "player.html", p=profile, matches=records(matches), scope=scope, other_seasons=sorted(others, reverse=True),
            first=engine.seasons[0], last=latest,
        )

    # ------------------------------------------------------------------ api
    @app.get("/api/health")
    def health():
        return {"status": "ok", "player_seasons": len(engine.table), "seasons": [engine.seasons[0], latest],
                "clusters": engine.model.k}

    @app.get("/api/players")
    def players():
        season = season_arg()
        t = engine.table if season is None else engine.table[engine.table["season"] == season]
        return jsonify(sorted(t["player"].unique().tolist()))

    @app.get("/api/archetypes")
    def archetypes():
        return jsonify(archetype_overview(season_arg() or latest))

    @app.get("/api/similar")
    def similar():
        name = request.args.get("player", "").strip()
        n = min(max(int(request.args.get("n", 10)), 1), 50)
        scope = request.args.get("scope", "season")
        if scope not in ("season", "all"):
            return jsonify(error="scope must be 'season' or 'all'"), 400
        idx, suggestions = lookup(name, season_arg())
        if idx is None:
            return jsonify(error=f"No qualifying season found for '{name}'", suggestions=suggestions), 404
        return jsonify(player=engine.player_profile(idx), similar=records(engine.similar(idx, n=n, scope=scope)))

    @app.post("/api/infer")
    def infer():
        body = request.get_json(silent=True) or {}
        stats = body.get("stats")
        if not isinstance(stats, dict) or not stats:
            return jsonify(error="Send {'stats': {feature: value, ...}, 'season': 2026}"), 400
        try:
            return jsonify(engine.infer(stats, int(body.get("season", latest)), n=int(body.get("n", 10))))
        except (KeyError, ValueError) as exc:
            return jsonify(error=str(exc)), 400

    # --------------------------------------------------------------- charts
    @lru_cache(maxsize=256)
    def radar_png(a: int, b: Optional[int]) -> bytes:
        ta = engine.table.loc[a]
        label_a = f"{ta['player']} ({ta['season']})"
        if b is None:
            return reports.radar(engine.Z[a], label_a)
        tb = engine.table.loc[b]
        return reports.radar(engine.Z[a], label_a, engine.Z[b], f"{tb['player']} ({tb['season']})")

    @lru_cache(maxsize=128)
    def map_png(idx: int, scope: str) -> bytes:
        row = engine.table.loc[idx]
        near = engine.similar(idx, n=5, scope=scope).index.tolist()
        season = int(row["season"]) if scope == "season" else None
        return reports.archetype_map(engine.table, engine.model.archetypes, highlight=[idx, *near], season=season)

    @app.get("/charts/radar.png")
    def radar():
        a, _ = lookup(request.args.get("player", ""), season_arg())
        if a is None:
            abort(404)
        vs = request.args.get("vs")
        vs_season = request.args.get("vs_season", "")
        b = engine.index.find(vs, int(vs_season) if vs_season.isdigit() else None) if vs else None
        return Response(radar_png(a, b), mimetype="image/png")

    @app.get("/charts/map.png")
    def map_chart():
        idx, _ = lookup(request.args.get("player", ""), season_arg())
        if idx is None:
            abort(404)
        scope = request.args.get("scope", "season")
        return Response(map_png(idx, "all" if scope == "all" else "season"), mimetype="image/png")

    @app.get("/reports/<path:filename>")
    def report_file(filename: str):
        return send_from_directory(report_dir, filename)

    return app
