#!/bin/sh

for s in {0..20}; do
  python simulation.py -seed $s
  python tidy_files.py
done
