"""Plot the Level of Stress map from lts_osm"""

import argparse
import logging
import numpy as np
import os
import pandas as pd
import geopandas as gpd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
    TimeRemainingColumn
)

logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler()]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plots the Level of Stress map calculated from lts_osm.py"
    )
    parser.add_argument(
        "--lts-csv-file",
        type=str,
        required=True,
        help="Path to the csv file of the LTS generated"
    )
    parser.add_argument(
        "--gdf-nodes-file",
        type=str,
        required=True,
        help="Path to the gdf nodes csv"
    )
    return parser.parse_args()

def main(args: argparse.Namespace) -> int:
    city = "Toronto"
    if not os.path.isfile(args.lts_csv_file):
        logger.error("Invalid --lts-csv-file path: %a", args.lts_csv_file)
        return 2
    if not os.path.isfile(args.gdf_nodes_file):
        logger.error("Invalid --gdf-nodes-file path: %a", args.gdf_nodes_file)
        return 2

    with open(args.lts_csv_file) as lts_csv_file:
        logger.info("Loading file %s", args.lts_csv_file)
        all_lts_df = pd.read_csv(lts_csv_file)
    with open(args.gdf_nodes_file) as gdf_nodes_file:
        logger.info("Loading file %s", args.gdf_nodes_file)
        gdfs_nodes = pd.read_csv(gdf_nodes_file, index_col=0)
    
    # convert to a geodataframe for ploting
    all_lts = gpd.GeoDataFrame(
        all_lts_df.loc[:, [c for c in all_lts_df.columns if c != "geometry"]],
        geometry=gpd.GeoSeries.from_wkt(all_lts_df["geometry"]),
        crs='wgs84'
    )

    # define lts colours for plotting
    conditions = [
        (all_lts['lts'] == 1),
        (all_lts['lts'] == 2),
        (all_lts['lts'] == 3),
        (all_lts['lts'] == 4)
    ]

    # create a list of the values we want to assign for each condition
    values = ['g', 'b', 'y', 'r']

    #create a new column and use np.select to assign values to it using our lists as arguments
    all_lts['color'] = np.select(conditions, values, default='none')

    lts_ranges = [
        (1, "LTS 1", "g"),
        (2, "LTS 2", "b"),
        (3, "LTS 3", "y"),
        (4, "LTS 4", "r"),
    ]

    fig, ax = plt.subplots()
    all_lts[all_lts['lts'] > 0].plot(ax = ax, linewidth= 0.1, color=all_lts[all_lts['lts'] > 0]['color'])
    handles = [Line2D([0], [0], color=c, lw=2, label=label) for _, label, c in lts_ranges]
    ax.legend(handles=handles, title="LTS", loc="upper right")
    plt.savefig(f"data/lts_{city}.pdf")
    plt.savefig(f"data/lts_{city}.png")

    ## Plot segments that aren't missing speed and lane info
    has_speed_lanes = all_lts[(~all_lts['maxspeed'].isna())
                              & (~all_lts['lanes'].isna())]
    
    fig, ax = plt.subplots(figsize = (8,8))
    has_speed_lanes.plot(ax=ax, linewidth=0.5, color=has_speed_lanes['color'])

    plt.savefig(f"LTS_{city}_has_speed_lanes.pdf")
    plt.savefig(f"LTS_{city}_has_speed_lanes.png", dpi=300)

    return 0

if __name__ == "__main__":
    raise SystemExit(main(parse_args()))