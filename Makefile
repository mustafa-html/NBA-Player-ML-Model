.PHONY: install train serve test fetch docker clean

install:
	pip install -r requirements-dev.txt

fetch:            ## refresh the season data from the source repo
	python -m playstyle.cli fetch

train:
	python -m playstyle.cli train --k 10

serve:
	FLASK_APP=wsgi:app flask run --debug --port 5000

test:
	pytest -q

docker:
	docker build -t nba-playstyle-engine . && docker run --rm -p 8000:8000 nba-playstyle-engine

clean:
	rm -rf artifacts reports .pytest_cache **/__pycache__
