import time
import csv
import copy
import sys
from profile_grids import grids as profile_grids
from solve_funcs import solve, check_solution
import matplotlib.pyplot as plt
from arg import (
    getArgs,
    TooManyHintsError,
    profiling,
    explain,
    to_file,
    barplot,
    hint,
)


def main() -> None:
    points = 0
    args = getArgs()
    if args.file:
        dims = {"4": (2, 2), "6": (2, 3), "9": (3, 3)}
        with open(args.file[0], "r", newline="") as gridfile:  # Context manager to open CSV file containing grid
            reader = csv.reader(gridfile)
            grid_input = [[int(value) for value in lst] for lst in list(reader)]
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
            except TooManyHintsError:  # Print error message and exit gracefully
                print("Error, number of hints requested is greater than the number of zeroes present in grid")
                sys.exit(1)
        if args.explain:
            changes = explain(grid_input, solution, False)
        else:
            changes = None
        to_file(args, solution, changes, grid_input)
    else:
        grids = copy.deepcopy(profile_grids)  # Create copy of profile_grids
        print("Running test script for coursework 3")
        print("====================================")
        for grid_number, (grid, n_rows, n_cols) in enumerate(grids):  #
            print("Solving grid: %d" % (grid_number + 1))
            start_time = (
                time.perf_counter()
            )  # time.perf_counter() instead of time.time() as we are interested in compute time rather than elapsed time
            solution = solve(grid, n_rows, n_cols, args.solver)
            elapsed_time = time.perf_counter() - start_time
            print("Solved in: %f seconds" % elapsed_time)
            if args.hint:
                try:
                    solution = hint(int(args.hint), solution, *profile_grids[grid_number])
                except TooManyHintsError:
                    print(f"Error, number of hints requested is greater than the number of zeroes present in grid {grid_number+1}")
                    sys.exit(1)
            print("\nOriginal Grid:")
            for line in profile_grids[grid_number][0]:  # Print unsolved grid to terminal
                print(line)
            print("\nSolution:")
            for line in solution:  # Print solved grid to terminal
                print(line)
            if check_solution(solution, n_rows, n_cols):
                print("grid %d correct\n" % (grid_number + 1))
                points = points + 10
            else:
                print("grid %d incorrect\n" % (grid_number + 1))
            if args.explain:
                explain(profile_grids[grid_number][0], solution, True)  # Print steps to reach solved grid to terminal
        if args.profile:
            repeats = 10  # Number of repeats per solver per Sudoku grid
            solvers = ["recursive", "wavefront"]  # Sudoku solvers implemented
            for solver in solvers:
                profiling_results = [profiling(grid, n_rows, n_cols, repeats, solver) for _, (grid, n_rows, n_cols) in enumerate(profile_grids)]
                barplot(profiling_results, repeats, solver)  # Create and show a bar plot detailing profiling results
            plt.show()  # Show profiling plots
        print("====================================")
        print("Test script complete, Total points: %d" % points)


if __name__ == "__main__":
    main()
