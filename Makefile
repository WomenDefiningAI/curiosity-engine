.PHONY: test demo

test:
	PYTHONPATH=src python -m unittest discover -s tests -v

demo:
	PYTHONPATH=src python scripts/bootstrap_demo.py
