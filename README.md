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
python main.py ARGUMENTS
```

## Arguments

| Parameter                 | Default       | Description   |
| :------------------------ |:-------------:| :-------------|
| --solver                  |   recursive   | Choose which solver to use {`recursive`, `wavefront`}|
| --explain                 |   False       | Explain changes between solution and input grid  |
| --hint N                  |   False       | Limit output to updating N values when showing solution - works with --explain and --file flags|
| --file INPUT OUTPUT       |   False       | Input a Sudoku grid from INPUT file and save output to OUTPUT file - works with --explain and --hint flags|
| --profile                 |   False       | Profile both the `recursive` and `wavefront` solvers against the grids in `profile_grids.py`. If selected all other flags are ignored.|

### Example Argument Usage

```bash
python main.py --file example.csv output.txt --hint 5 --explain --solver wavefront
```

## Testing other Sudoku grids

When testing individual Sudoku grids from a CSV file, the `--file` argument should be used. Sudoku grids should be provided in the same format as shown in `example.csv`, where newlines delineate rows and commas delineate columns.

To test multiple Sudoku grids at once for Tasks 1 and 3, `default_grids.py` can be updated with additional grids in a nested list format. The `--solver` flag can be used to select either the `recursive` or `wavefront` solver. Each list should contain a row of values. Subsequently, the `grids` variable should be updated with the variable containing a tuple where the first element is the variable containing the nested list for the grid, and the second and third elements are the dimension of the Sudoku grid. 
