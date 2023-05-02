# CW3

## Charles Harris, Jack Cooper, Owen Parsonage, and Ben Joseph

This repository contains a Sudoku solver written in Python. Two solving algorithms are included, `recursive` and `wavefront`.

## Arguments

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| --solver                  |   recursive   | Choose which solver to use {`recursive`, `wavefront`}|
| --explain                 |   False       | Explain changes between solution and input grid  |
| --hint N                  |   False       | Limit output to updating N values when showing solution - works with --explain and --file flags|
| --file INPUT OUTPUT       |   False       | Input a Sudoku grid from INPUT file and save output to OUTPUT file - works with --explain and --hint flags|
| --profile                 |   False       | Profile both the `recursive` and `wavefront` solvers against the grids in `profile_grids.py`|