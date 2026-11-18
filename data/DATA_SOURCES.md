# Data sources

All raw files in `data/raw/` are kept exactly as published by their source (same separators,
same codes, same markers for missing values). `ddm.load_dataset()` only parses them; all
cleaning and encoding happens in the notebooks.

## penguins - Palmer Archipelago penguins

| | |
|---|---|
| Source | https://allisonhorst.github.io/palmerpenguins/ (repository: https://github.com/allisonhorst/palmerpenguins) |
| Licence | CC0 1.0 (public domain dedication) |
| Citation | Horst AM, Hill AP, Gorman KB (2020). palmerpenguins: Palmer Archipelago (Antarctica) penguin data. R package. doi:10.5281/zenodo.3960218 |
| Original data | Collected by Dr. Kristen Gorman and the Palmer Station Long Term Ecological Research (LTER) program |
| Raw file | `penguins.csv` - the simplified `penguins` table of the R package |

`penguins.csv`: comma-separated, header line, 344 rows. Columns `species` (Adelie, Chinstrap,
Gentoo), `island` (Biscoe, Dream, Torgersen), `bill_length_mm`, `bill_depth_mm`,
`flipper_length_mm`, `body_mass_g`, `sex` (female, male), `year` (2007-2009). Missing values are
written as `NA` (2 rows without any measurement, 11 rows without `sex`).

## wholesale - Wholesale customers

| | |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/292/wholesale+customers |
| Licence | CC BY 4.0 |
| Citation | Cardoso, M. (2013). Wholesale customers [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5030X |
| Raw file | `wholesale_customers.csv` |

`wholesale_customers.csv`: comma-separated, header line, 440 rows, no missing values. One row
per client of a wholesale distributor in Portugal. `Channel` (1 = Horeca: hotel/restaurant/café,
2 = Retail) and `Region` (1 = Lisbon, 2 = Oporto, 3 = other) are nominal attributes stored as
integer codes. `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen` are the
annual spending in monetary units per product category (the loader renames the columns to lower
case and `Delicassen` to `delicatessen`).

## auto_mpg - Auto MPG

| | |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/9/auto+mpg |
| Licence | CC BY 4.0 |
| Citation | Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5859H |
| Raw files | `auto-mpg.data`, `auto-mpg.names` (original description) |

`auto-mpg.data`: whitespace-separated, **no header**, 398 rows, the car name in double quotes at
the end of each line. Columns in order: `mpg`, `cylinders`, `displacement`, `horsepower`,
`weight`, `acceleration`, `model_year` (70-82, i.e. 1970-1982), `origin` (1 = USA, 2 = Europe,
3 = Japan), `car_name`. Missing values are written as `?` (6 times in `horsepower`).

## students - Student performance (math course)

| | |
|---|---|
| Source | https://archive.ics.uci.edu/dataset/320/student+performance |
| Licence | CC BY 4.0 |
| Citation | Cortez, P. (2008). Student Performance [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T |
| Paper | Cortez, P., Silva, A. (2008). Using data mining to predict secondary school student performance. Proc. 5th FUture BUsiness TEChnology Conference (FUBUTEC 2008), 5-12. |
| Raw files | `student-mat.csv`, `student.txt` (attribute description) |

`student-mat.csv`: **semicolon**-separated, header line, 395 rows, no missing values. Students
of the math course at two Portuguese secondary schools. 33 attributes of every type: binary
(e.g. `school`, `sex`, `schoolsup`), nominal (`Mjob`, `Fjob`, `reason`, `guardian`), ordinal codes
(e.g. `Medu`, `studytime`, `famrel` on 1-5 scales) and numeric (`age`, `absences`, grades
`G1`, `G2`, `G3` on 0-20). The meaning of every code is in `student.txt`. Only the math file of the
original dataset is included (the Portuguese course file `student-por.csv` is not).
