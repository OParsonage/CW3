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
class ProfileResults:
    """Class for storing parameters to plot profiling results"""

    difficulty: int
    n_rows: int
    n_cols: int
    timeit_results: list


def _getArgs() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        help="Measure performance and produce plots using 'timeit'",
        default=False,
        action=argparse.BooleanOptionalAction,
    )

    args = parser.parse_args()

    if args.profile:
        args.explain, args.file, args.hint = False, False, False
    return args


def reader(
    data_file: str,
) -> list[list[int]]:  # function to read the data from the csv file
    with open(data_file) as data:
        all_data = csv.reader(data)  # read the data
        return list(all_data)  # return the data as a list


def explain(
    original_grid: list[list[int]],
    solved_grid: list[list[int]],
    to_terminal: bool,
) -> list[list[tuple[int, int]]]:
    changes = []
    for index, row in enumerate(original_grid):
        changes.append(
            [
                (i, updated)
                for i, (zero, updated) in enumerate(
                    zip(row, solved_grid[index])
                )
                if zero != updated
            ]
        )
    if to_terminal:
        for row_number, row in enumerate(changes):
            for element in row:
                print(
                    f"Put {element[0]} in location ({row_number}, {element[1]})"
                )
    return changes


def to_file(
    args: argparse.Namespace,
    solved_grid: list[list[int]],
    changes: list[list[tuple[int]]],
) -> None:
    with open(args.file[1], "w") as output:
        output.write("Solved Grid:\n")
        writer = csv.writer(output)
        writer.writerows(solved_grid)
    if args.explain:
        with open(args.file[1], "a") as output:
            output.write("\n\nExplanation:\n")
            for row_number, row in enumerate(changes):
                for element in row:
                    output.write(
                        f"Put {element[0]} in location ({row_number}, {element[1]})\n"
                    )


def hint(
    hints: int,
    solved_grid: list[list[int]],
    original_grid: list[list[int]],
    n_rows: int,
    n_cols: int,
):
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


# function to plot the results of the profiling as a bar chart


def profiling(
    grid: list[list[int]], n_cols: int, n_rows: int, repeat: int, solver: str
) -> ProfileResults:
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

    return ProfileResults(difficulty, n_rows, n_cols, results)


def barplot(results: list[ProfileResults], repeats: int) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Time taken to solve Sudoku puzzles")
    ax.set_xlabel("Number of missing values")
    ax.set_ylabel("Time taken (s)")
    ax.set_xticks(range(0, 81, 5))
    ax.set_yticks(np.arange(0, 2.5, 0.25))
    ax.grid(True)

    # Define color map for difficulty levels
    cmap = plt.get_cmap("rainbow")

    # sorting the grids by difficulty
    results.sort(key=lambda x: x.difficulty)

    grid_number = 0
    for grid in results:  # we plot the results for each grid
        grid_number += 1
        difficulty = grid.difficulty  # we get the difficulty of the grid
        average_time = np.mean(
            grid.timeit_results
        )  # we calculate the average time taken to solve the grid
        std_dev = np.std(
            grid.timeit_results
        )  # we calculate the standard deviation of the time taken to solve the grid

        # Adding error bars to bar plot
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

    # Add subtitle with additional information about the data
    fig.text(
        0.5,
        0.001,
        f"Data based on 18 Sudoku puzzles solved {repeats} times each using a recursive algorithm",
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

    # sorting x and y values by ascending y values
    x = x[np.argsort(x)]
    y = y[np.argsort(x)]

    # sorting x and y values by ascending y values
    x = x[np.argsort(x)]
    y = y[np.argsort(x)]

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
    plt.show()  # we show the plot
