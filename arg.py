import argparse
import numpy as np
import csv
import itertools
import random
import copy
import timeit
import matplotlib.pyplot as plt
from dataclasses import dataclass
from solve_funcs import solve


class TooManyHintsError(Exception):
    """Raised when user asks for more hints than there are zeroes in Sudoku grid"""

    pass


@dataclass
class ProfileResult:
    """Class for storing parameters to plot profiling results."""

    difficulty: int
    n_rows: int
    n_cols: int
    timeit_results: list


def _getArgs() -> argparse.Namespace:
    """
    Create and parse arguments from main.py.
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--solver",
        choices=["recursive", "wavefront"],
        help="Choose either recursive or wavefront solver",
        default="recursive",
    )
    parser.add_argument(
        "--explain",
        help="Provide set of instructions for solving grid",
        default=False,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--file",
        nargs=2,
        metavar=("INPUT", "OUTPUT"),
        help="Input file for grid to solve and output file for solved grid. If 'explain' is set then include explanations",
    )
    parser.add_argument(
        "--hint",
        help="Integer value for number of values to fill",
        default=None,
    )
    parser.add_argument(
        "--profile",
        help="Measure performance and produce plots using 'timeit'. Other arguments will be ignored if this is set",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    if args.profile:
        args.explain, args.file, args.hint = False, False, False
    return args


def reader(
    data_file: str,
) -> list[list[int]]:
    """
    Function to read grid input from a file in a CSV format.

    Args:
        data_file (str): Relative path to grid CSV input file

    Returns
        list[list[int]]: Nested list containing the grid
    """
    with open(data_file) as data:  # Context manager to handle file
        all_data = csv.reader(data)  # Read CSV grid
        return list(all_data)  # Return the grid as a nested list


def explain(
    original_grid: list[list[int]],
    solved_grid: list[list[int]],
    to_terminal: bool,
) -> list[list[tuple[int, int]]]:
    """
    Function to determine the differences between the original and solved grids.

    Args:
        original_grid (list[list[int]]): Unsolved grid
        solved_grid (list[list[int]]): Solved grid
        to_terminal (bool): If True, print to terminal

    Returns:
        changes (list[list[tuple[int, int]]]): Nested list containing the changes between the original and solved grids at each point
    """
    changes = [
        [
            (i, updated)
            for i, (zero, updated) in enumerate(zip(row, solved_grid[index]))
            if zero != updated
        ]
        for index, row in enumerate(original_grid)
    ]
    if to_terminal:  # False when --explain flag is set
        for row_number, row in enumerate(changes):
            for element in row:
                print(
                    f"Put {element[1]} in row {row_number+1}, column {element[0]+1}"
                )
    return changes


def to_file(
    args: argparse.Namespace,
    solved_grid: list[list[int]],
    changes: list[list[tuple[int]]],
    original_grid: list[list[int]],
) -> None:
    """
    Writes output to file. If --explain flag is set, include this in output file.
    If --hint N flag is set limit changes between original and solved grids to N.

    Args:
        args (argparse.Namespace): Parsed Arguments
        solved_grid (list[list[int]]): Solved grid
        changes (list[list[tuple[int, int]]]): Nested list containing the changes between the original and solved grids at each point
        original_grid (list[list[int]]): Unsolved grid
    """
    with open(args.file[1], "w") as output:
        writer = csv.writer(output)
        output.write("Original Grid:\n")
        writer.writerows(original_grid)
        output.write("\n")
        output.write("Solved Grid:\n")
        writer.writerows(solved_grid)
    if args.explain:
        with open(args.file[1], "a") as output:
            output.write("\n\nExplanation:\n")
            for row_number, row in enumerate(changes):
                for element in row:
                    output.write(
                        f"Put {element[1]} in row {row_number+1}, column {element[0]+1}\n"
                    )


def hint(
    hints: int,
    solved_grid: list[list[int]],
    original_grid: list[list[int]],
    n_rows: int,
    n_cols: int,
) -> None:
    """
    When --hint N flag is set, show N changes between the input and solved grid. When --explain flag is set show the requisite steps to reach the grid with N changes.

    Args:
        hints (int): Number of hints to show
        solved_grid (list[list[int]]): Solved grid
        original_grid (list[list[int]]): Unsolved grid
        n_rows (int): Number of rows in grid
        n_cols (int): Number of columns in grid
    """
    ranges = [range(0, n_rows * n_cols) for i in range(2)]
    perms = list(itertools.product(*ranges))
    valid_perms = [
        perm for perm in perms if original_grid[perm[0]][perm[1]] == 0
    ]
    if hints > sum(row.count(0) for row in original_grid):
        raise TooManyHintsError
    chosen_hints = random.sample(valid_perms, hints)
    grid_to_show = copy.deepcopy(original_grid)
    for perm in chosen_hints:
        grid_to_show[perm[0]][perm[1]] = solved_grid[perm[0]][perm[1]]
    return grid_to_show


def profiling(
    grid: list[list[int]], n_cols: int, n_rows: int, repeat: int, solver: str
) -> ProfileResult:
    """
    Profile solver against grid argument. Runs solver against grid for the number of repeats specified in the 'repeat' argument.

    Arguments:
        grid (list[list[int]]): Unsolved grid
        n_cols (int): Number of columns in grid
        n_rows (int): Number of rows in grid
        repeat (int): Number of repeats
        solver (str): Solver to use

    Returns:
        ProfileResult: Dataclass containing profiling results for a single grid
    """
    SETUP = """
import copy
grid_to_test = copy.deepcopy(grid)
"""  # Deepcopy required to prevent mutation of grid variable for subsequent runs. Setup code is not included in execution time.

    STMT = """
solved_grid = solve(grid_to_test, n_rows, n_cols, solver)
"""

    difficulty = sum(row.count(0) for row in grid)
    results = timeit.repeat(
        stmt=STMT,
        setup=SETUP,
        repeat=repeat,
        number=1,
        globals={
            "grid": grid,
            "n_rows": n_cols,
            "n_cols": n_rows,
            "solve": solve,
            "solver": solver,
        },
    )

    return ProfileResult(difficulty, n_rows, n_cols, results)


def barplot(results: list[ProfileResult], repeats: int, solver: str) -> None:
    """
    Produce a bar plot showing profiling results.

    Args:
        results (list[ProfileResult]): List of profiling result dataclasses
        repeats (int): Number of repeats that profiling was run for
        solver (str): Solver used to produce profiling results
    """
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title(
        f"Time taken to solve Sudoku puzzles for {solver.capitalize()} solver"
    )
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)

    # Define color map for difficulty levels
    cmap = plt.get_cmap("rainbow")

    # Sorting the grids by difficulty
    results.sort(key=lambda x: x.difficulty)

    grid_number = 0
    for grid in results:  # Calculate error bars and create a bar for each plot
        grid_number += 1
        difficulty = grid.difficulty  # Difficulty of each grid
        average_time = np.mean(
            grid.timeit_results
        )  # Calculate average time to solve each grid
        std_dev = np.std(
            grid.timeit_results
        )  # Calculate standard deviation for time to solve each grid

        ax.bar(
            difficulty,
            average_time,
            yerr=std_dev,
            capsize=5,
            color=cmap(difficulty / 80),
            label=f"Grid {grid_number}: {grid.n_rows}x{grid.n_cols}",
        )

    # Increase font size of axis labels
    ax.tick_params(axis="both", labelsize=12)

    # Set y-axis to logarithmic scale
    ax.set_yscale("log")
    ax.set_ylim(10e-6, 10e-1)

    # Add subtitle with additional information about the data
    fig.text(
        0.5,
        0.001,
        f"Data based on {len(results)} Sudoku puzzles solved {repeats} times each using a recursive algorithm",
        ha="center",
        fontsize=12,
    )

    # Add color bar to indicate difficulty levels
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=80))
    cbar = fig.colorbar(sm, ax=ax, ticks=range(0, 81, 10), shrink=0.8)
    cbar.set_label("Difficulty", fontsize=12)

    # Extract the x and y values from the results
    x = np.array([grid.difficulty for grid in results])
    y = np.array([np.mean(grid.timeit_results) for grid in results])

    # Fit a logarithmic polynomial to the data
    z = np.polyfit(np.log(x), np.log(y), 1)
    f = np.poly1d(z)

    # Plot the logarithmic line of best fit
    ax.plot(
        x,
        np.exp(f(np.log(x))),
        color="black",
        linestyle="--",
        label="Line of best fit (log-log)",
    )
    # Add legend
    plt.legend(bbox_to_anchor=(1.2, 1), loc="upper left")
    plt.tight_layout()
