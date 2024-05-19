## NHFLO models
Note that the content of this repository is available under a not-so-permissive open-source licence: GNU AGPLv3. Please have a look at [choose a license](https://choosealicense.com/licenses/agpl-3.0/) for the key conditions and limitations of this license before getting started.

Get started directly with a preconfigured modeling environment using [GitHub's Codespace](https://github.com/codespaces/new?hide_repo_select=true&ref=main&repo=758399015&skip_quickstart=true&machine=standardLinux32gb&devcontainer_path=.devcontainer%2Fdevcontainer.json&geo=EuropeWest). This is by far the easiest method to get everything started, else follow the instructions for other platforms below.

| Modelscript folder | Notebooks up and running | Pinned or rolling versions | Format |
| --- | --- | --- | --- |
| 09pwnmodel2 | [![09pwnmodel2 tests](https://github.com/NHFLO/models_public/actions/workflows/Tests_09pwnmodel2.yml/badge.svg)](https://github.com/NHFLO/models_public/actions/workflows/Tests_09pwnmodel2.yml) | Rolling | [![Format of 09pwnmodel2](https://github.com/NHFLO/models_public/actions/workflows/Format_09pwnmodel2.yml/badge.svg)](https://github.com/NHFLO/models_public/actions/workflows/Format_09pwnmodel2.yml)|

The production modelscripts should be running and green. The modelscripts annotated as work in progress are likely to be failing and red. Formatting should be perfect.

### Basic installation
- Install git (https://gitforwindows.org/)
- Install GitHub desktop (https://desktop.github.com/)
- Install VS Code (https://code.visualstudio.com/)
  - Open VScode and install the following extensions: Python, Jupyter, GitHub
  - Close VScode
- Install Anaconda (https://www.anaconda.com/download). During installation make the following changes to the default options: Install for user, Add to Path, Make default Python for programs like VS code
- Use GitHub Desktop to clone github.com/nhflo/models to a local directory, e.g., "C:\Users\tombb\Python Scripts\NHFLO\models"
  - Don't use a folder on your OneDrive/Google Drive.
  - Place it in a parent directory called NHFLO.
- Open Anaconda prompt
  - Run: `conda config --append channels conda-forge`
  - Run: `conda create --name nhflobase python=3.11 "hatch>=1.11.0" "setuptools>=64.0.0"`
  - Run: `conda activate nhflobase`
  - Run: `hatch config set dirs.env.virtual .direnv`
  - Done. You can now run via `hatch` the notebooks in Jupyter, Visual Code, Spyder, or Pycharm following the steps outlined below.

### Running the notebooks in Jupyter
The folder modelscripts contains folders with notebooks to create groundwater models. Each of these folders contain notebooks that work with the same dependencies, such as the same version of `nlmod` and the same version of `nhflodata`. Such a collection of dependencies is called an environment. The notebooks require to be run using the environment that is specified in the folder name. For example, the folder `09pwnmodel2` contains notebooks that can be run using the environment `09pwnmodel2`. Run Jupyter configured to run notebooks from the `./modelscripts/09pwnmodel2` folder using the following steps:
- Open Anaconda prompt and navigate to the `models` repository folder you just cloned:
  - Run: `conda activate nhflobase`
  - Run: `cd "C:\Users\tombb\Python Scripts\NHFLO\models"`
  - Run: `hatch run 09pwnmodel2:jnb`

Replace `09pwnmodel2` with the project folder of choice.

### Running the notebooks in Visual Code
The following command creates the environment of which the kernel can be selected for the notebooks:
- Open Anaconda prompt and navigate to the `models` repository folder you just cloned:
  - Run: `conda activate nhflobase`
  - Run: `cd "C:\Users\tombb\Python Scripts\NHFLO\models"`
  - Run: `hatch env create 09pwnmodel2`

The path to the environment is `... models\.direnv\09pwnmodel2\`, as per the hatch config command in the installation instructions, which is automatically added to the search path for Python interpreters. This means that you can select the environment in Visual Code, Spyder, or Pycharm.

Start VSCode and open the models folder: 
- File > Open Folder.. > "C:\Users\tombb\Python Scripts\NHFLO\models"

### Developer instructions
#### Local development
If you are also editing NHFLO/data, NHFLO/tools, or github.com/gwmod/nlmod content you will want to clone those repositories and install the content as an editable package. The steps below outline the steps to create a separate `localdev` environment with those packages installed.
- Open GitHub Desktop > Clone:
  - [data](https://github.com/NHFLO/data) to "C:\Users\tombb\Python Scripts\NHFLO\data"
  - [tools](https://github.com/NHFLO/tools) to "C:\Users\tombb\Python Scripts\NHFLO\tools"
  - [nlmod](https://github.com/gwmod/nlmod) to "C:\Users\tombb\Python Scripts\nlmod"
  - [hydropandas](https://github.com/ArtesiaWater/hydropandas) to "C:\Users\tombb\Python Scripts\hydropandas"
- Open Anaconda prompt and navigate to the `models` repository folder you just cloned:
  - Run: `conda activate nhflobase`
  - Run: `cd "C:\Users\tombb\Python Scripts\NHFLO\models"`
  - Run: `hatch env create localdev`
- Open VSCode and add those local folders to your workspace.
- Use the localdev environment to run your notebooks and python scripts.

#### Testing
After making changes to successfully the notebooks need to pass the tests. Run the tests using the following command:
```bash
hatch run 09pwnmodel2:test
```

#### Commandline
In the case that you want to install an additional package in an environment or perform other commands from the commandline, you need to enter the shell. To enter the shell with a specific environment select use the following command
```bash
hatch -e 09pwnmodel2 shell
pip install additional_package
```
also you can run `pip`, `jupyter`, etc.

#### Formatting
After passing the tests the formatting of the code needs to match the quality standards. Running the `lint`-command will show the formatting errors. Running the `format`-command will fix the formatting errors that can be fixed automatically. The remaining errors need to be fixed manually. Linting and formatting the code is done using tools available in the `lintformat` environment.
```bash
hatch run lintformat:lint
hatch run lintformat:format
```
