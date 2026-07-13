#  (C) Crown Copyright, Met Office, 2026
from pathlib import Path
import os
def main():
    print("running analysis")
    isca_datadir = Path(os.getenv("GFDL_DATA"))
    print(f"{isca_datadir=}")
    exp_datadir = isca_datadir.joinpath("soc_aquaplanet", "run_0001")


if __name__=="__main__":
    main()
