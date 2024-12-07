# Supervised-Feature-Selection-and-Extraction
This repository implements various feature selection and extraction methods to reduce the dimensionality of datasets from m features to 2 features. It includes Python scripts that perform tasks such as minimizing and maximizing scatter metrics, as well as supervised feature selection using criteria like Pearson correlation and Fisher's criterion. # Supervised Feature Selection and Extraction

## Project Overview

This project implements various dimensionality reduction techniques to transform high-dimensional data into 2D representations through feature selection and extraction methods.

## Data Format

### Input

- **Data File**: n×m matrix (comma-separated)
  - n = number of data items (rows)
  - m = number of features (columns)
- **Labels File**: Array of n numeric values

### Output

- Reduced data file: n×2 matrix (comma-separated)

## Implemented Programs

### Scatter-based Methods

1. `scatter1.py` - Mixture-class scatter minimization
1. `scatter2.py` - Mixture-class scatter maximization
1. `scatter3.py` - Within-class scatter minimization
1. `scatter4.py` - Within-class scatter maximization
1. `scatter5.py` - Between-class scatter minimization
1. `scatter6.py` - Between-class scatter maximization
1. `scatter7.py` - Between-class/within-class scatter ratio minimization
1. `scatter8.py` - Between-class/within-class scatter ratio maximization

### Selection Methods

1. `selection1.py` - Pearson correlation criterion selection
1. `selection2.py` - Fisher criterion selection
1. `selection3.py` - Recursive elimination with linear discriminant

## Usage Instructions

### Running Programs

All programs follow the same command-line format:

```
python3 program.py datafile labelsfile outputfile
```

Example:

```
python3 scatter1.py irisdata.txt irislabels.txt iris2d.txt
```
<br>
