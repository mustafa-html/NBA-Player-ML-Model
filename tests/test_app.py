def test_first_run_trains_into_the_configured_directories(tmp_path):
    """A fresh install has no artifacts; create_app must build them where told."""
    from webapp import create_app

    app = create_app(tmp_path / "artifacts", tmp_path / "reports")
    app.testing = True
    c = app.test_client()
    assert c.get("/api/health").get_json()["status"] == "ok"
    assert c.get("/reports/playstyle_map.png").status_code == 200


def test_health(client):
    r = client.get("/api/health").get_json()
    assert r["status"] == "ok" and r["clusters"] == 10 and r["seasons"] == [2016, 2026]


def test_pages_render(client):
    assert client.get("/").status_code == 200
    r = client.get("/player?name=luka+doncic&season=2026")
    assert r.status_code == 200 and "Luka Dončić" in r.get_data(as_text=True)
    assert client.get("/player?name=Luka+Doncic&season=2026&scope=all").status_code == 200


def test_unknown_player_suggests(client):
    r = client.get("/player?name=Lebron")
    assert r.status_code == 404 and "LeBron James" in r.get_data(as_text=True)
    j = client.get("/api/similar?player=Lebron").get_json()
    assert "LeBron James" in j["suggestions"]


def test_similar_api(client):
    j = client.get("/api/similar?player=Stephen Curry&season=2016&n=5&scope=all").get_json()
    assert j["player"]["player"] == "Stephen Curry" and len(j["similar"]) == 5
    assert all(m["player"] != "Stephen Curry" for m in j["similar"])
    assert client.get("/api/similar?player=Stephen Curry&scope=bad").status_code == 400


def test_infer_api(client):
    stats = {"rim": 0.45, "three": 0.02, "dunk": 0.30, "orb": 13.0, "blk": 5.0, "usg": 14.0, "astp": 6.0}
    j = client.post("/api/infer", json={"stats": stats, "season": 2026, "n": 3}).get_json()
    assert j["archetype"] in {"Rim-Running Big", "Glass Cleaner"}
    assert len(j["similar"]) == 3
    assert client.post("/api/infer", json={"stats": {}}).status_code == 400
    assert client.post("/api/infer", json={"stats": stats, "season": 1990}).status_code == 400


def test_charts_are_png(client):
    for url in ("/charts/radar.png?player=Nikola Jokic&season=2026&vs=Domantas Sabonis&vs_season=2026",
                "/charts/map.png?player=Nikola Jokic&season=2026", "/reports/archetype_trend.png"):
        r = client.get(url)
        assert r.status_code == 200 and r.data[:4] == b"\x89PNG", url
