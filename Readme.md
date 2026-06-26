# Quantum Hall effect


## IDE setup

### VSCode

- Recommended extensions
	- LaTeX Workshop (LaTeX compilation)
	- Pylance


## Code setup

### Conda environment

Create the `conda` environment
```sh
conda env create -f environment.yml
```

Changing the environment
`--prune` removes packages which are not needed
```sh
conda env update -f environment.yml --prune
```

Activate the environment
```sh
conda activate kwant-env
```

### Run the python code

Activate the conda environment
```sh
conda activate kwant-env
```

Run
```sh
python3 hall_resistivity.py
```


## $\LaTeX$

### Compilation of `.tex` files

- In terminal
	```sh
	pdflatex KwantQHE.tex --output-directory=LatexOut
	```
- In VSCode with the *LaTeX Workshop* extension:
	- For any open `.tex` file
		- Automatic: Change and save the file, e.g., with `CMD + S`.
		- Manual: Press the green 'play button' in the top right corner.

## Questions

- Should the Hall bar have equal chemical potential in the scattering system and the leads?
	- Zeb: The scattering system has a gate on top of it which fixes the density of electrons in the scattering system, but not the leads.
		Thus, the chemical potential in the scattering region changes as a function magnetic filed, but the leads don't.