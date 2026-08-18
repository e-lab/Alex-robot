ISAACLAB ?= /home/pacquadr/IsaacLab/isaaclab.sh
PACKAGE_PATH := src
RUNNER := scripts/validate/run_validation.py
RUFF_PATHS := scripts $(PACKAGE_PATH) tests

.PHONY: test test-gpu test-release

test:
	ruff check .
	ruff format --check $(RUFF_PATHS)
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p scripts/generate/generate_alex_purdue.py --check
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p scripts/generate/generate_weiss_wsg32.py --check
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p scripts/generate/prepare_zed_x_mini.py --check
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p scripts/generate/generate_realsense.py --check
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p scripts/generate/generate_leap_hand_v1.py --check
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p -m pytest -q tests

test-gpu:
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p $(RUNNER) --component "$(COMPONENT)"

test-release: test
	PYTHONPATH=$(PACKAGE_PATH) $(ISAACLAB) -p $(RUNNER)
