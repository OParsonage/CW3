# CW3

## Charles Harris, Jack Cooper, Owen Parsonage, and Ben Joseph

This repository contains a Sudoku solver written in Python. Two solving algorithms are included, `recursive` and `wavefront`.

Note: Python 3.10+ required due to the use of PEP 604 union type hinting.

## Dependencies

A Python environment with the following dependencies is required:

- [matplotlib](https://pypi.org/project/matplotlib/)
- [numpy](https://pypi.org/project/numpy/)

## Example Usage

```bash
python main.py
```

## Arguments

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| --solver                  |   recursive   | Choose which solver to use {`recursive`, `wavefront`}|
| --explain                 |   False       | Explain changes between solution and input grid  |
| --hint N                  |   False       | Limit output to updating N values when showing solution - works with --explain and --file flags|
| --file INPUT OUTPUT       |   False       | Input a Sudoku grid from INPUT file and save output to OUTPUT file - works with --explain and --hint flags|
| --profile                 |   False       | Profile both the `recursive` and `wavefront` solvers against the grids in `profile_grids.py`. If selected all other flags are ignored.|

For use with the `--file` argument, Sudoku grids should be provided in the same format as shown in `example.csv`, where newlines delineate rows and commas delineate columns.
