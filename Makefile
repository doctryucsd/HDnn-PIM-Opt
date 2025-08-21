# Makefile
# Usage:
#   make run                # runs with defaults N=4, START=0
#   make run N=8 START=100  # override defaults
#   make clean              # remove outputs/ and dbg*.txt logs

SHELL := /bin/bash

# Config
DRIVE  ?= /cimloop-volume
N      ?= 10        # number of instances
START  ?= 0        # starting number added to seed offset
PYTHON ?= python3  # python interpreter

.PHONY: all run clean

all: run

run:
	@echo "Starting $(N) sim processes (START=$(START))..."
	@for ((x=0; x<$(N); x++)); do \
	  ( \
	    seed=$$((42 + $(START) + x)); \
	    echo "Launching seed $$seed"; \
	    $(PYTHON) -m sim seed="$$seed" > "$(DRIVE)/dbg$$seed.txt" 2>&1; \
	  ) & \
	  echo "Started sim process $$x"; \
	  sleep 5; \
	done; \
	wait; \
	echo "All $(N) sim processes completed."

clean:
	@echo "Cleaning outputs/ and $(DRIVE)/dbg*.txt ..."
	-@rm -rf outputs/
	-@rm -f "$(DRIVE)"/dbg*.txt
