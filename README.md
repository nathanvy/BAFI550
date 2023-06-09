This repository contains python version 3 source code to generate plots for a term paper for BAFI 550 at the UBC Sauder School of Business.  You can generate the plots by cloning the repository and issuing the `make` command on any POSIX-compliant system.

Requires Python 3.10+ and the following libraries:
* yFinance, for downloading market data
* Scikit-Learn, for running linear regressions
* Matplotlib, for plotting
* NumPy, for some array operations
* Seaborn, for fancy-shmancy plots
* Pandas, for data frame operations

Stock data is fetched from the Yahoo Finance API except for TFCF, which has been delisted from Yahoo after the Disney merger and so that data is stored in a comma-delimited file, `tfcf.csv`, retrieved from MarketWatch in May of 2023.
