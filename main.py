import time
import csv
import copy
import sys
from original_grids import grids as original_grids
from solve_funcs import solve, _check_solution
from arg import (
    _getArgs,
    TooManyHintsError,
    profiling,
    explain,
    to_file,
    barplot,
    hint,
)


def _main() -> None:
    points = 0
    args = _getArgs()
    if args.file:
        dims = {"4": (2, 2), "6": (2, 3), "9": (3, 3)}
        with open(args.file[0], "r", newline="") as gridfile:
            reader = csv.reader(gridfile)
            grid_input = [
                [int(value) for value in lst] for lst in list(reader)
            ]
            solution = solve(
                copy.deepcopy(grid_input),
                *dims[str(len(grid_input))],
                args.solver,
            )
        if args.hint:
            try:
                solution = hint(
                    int(args.hint),
                    solution,
                    grid_input,
                    *dims[str(len(grid_input))],
                )
            except TooManyHintsError:
                print(
                    "Error, number of hints requested is greater than the number of zeroes present in grid"
                )
                sys.exit(1)
        if args.explain:
            changes = explain(grid_input, solution, False)
        else:
            changes = None
        to_file(args, solution, changes)
    else:
        grids = copy.deepcopy(original_grids)
        print("Running test script for coursework 1")
        print("====================================")
        for i, (grid, n_rows, n_cols) in enumerate(grids):
            print("Solving grid: %d" % (i + 1))
            start_time = time.time()
            solution = solve(grid, n_rows, n_cols, args.solver)
            elapsed_time = time.time() - start_time
            print("Solved in: %f seconds" % elapsed_time)
            if args.hint:
                try:
                    solution = hint(
                        int(args.hint), solution, *original_grids[i]
                    )
                except TooManyHintsError:
                    print(
                        f"Error, number of hints requested is greater than the number of zeroes present in grid {i+1}"
                    )
                    sys.exit(1)
            for line in solution:
                print(line)
            if _check_solution(solution, n_rows, n_cols):
                print("grid %d correct" % (i + 1))
                points = points + 10

            else:
                print("grid %d incorrect" % (i + 1))
            if args.explain:
                explain(original_grids[i][0], solution, True)
        if args.profile:
            repeats = 10
            profiling_results = [
                profiling(grid, n_rows, n_cols, repeats, args.solver)
                for _, (grid, n_rows, n_cols) in enumerate(original_grids)
            ]
            barplot(profiling_results, repeats)

        print("====================================")
        print("Test script complete, Total points: %d" % points)


if __name__ == "__main__":
    _main()
