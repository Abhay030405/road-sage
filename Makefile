.PHONY: install test test-integration lint lint-fix api dashboard \
        docker-up docker-down evaluate ablation stress-test phase7-check all

install:
	pip install -r requirements.txt

test:
	pytest tests/ -v --tb=short

test-integration:
	pytest tests/test_integration.py -v --tb=short

lint:
	ruff check .

lint-fix:
	ruff check . --fix

api:
	uvicorn api.main:app --reload --port 8000

dashboard:
	cd dashboard && npm run dev

docker-up:
	docker-compose up --build

docker-down:
	docker-compose down

evaluate:
	python training/evaluation/evaluate_lane.py --source rgb
	python training/evaluation/evaluate_decision.py --source rgb --benchmark

ablation:
	python training/scripts/run_ablation.py --source rgb --n-images 50

stress-test:
	python training/scripts/metrics_stress_test.py --duration-minutes 2

phase7-check:
	python training/evaluation/evaluate_phase7.py

all: install test lint evaluate
