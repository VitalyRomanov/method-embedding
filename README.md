# Source Code Tools
Library for analyzing source code with graphs and NLP. What this repository can do:

1. Fetch source codes for packages in pip
2. Create indexes of python packages using [Sourcetrail](https://www.sourcetrail.com)
3. Convert Sourcetrail indexes into a connected graph
4. Build graphs for source codes from AST
5. Train Graph Neural Network for learning representations for source code
6. Predict Python types using NLP and graph embeddings

### Installation

You need to use conda, create virtual environment `SourceCodeTools` with python 3.8
```bash
conda create -n SourceCodeTools python=3.8
```

If you plan to use graphviz
```python
conda install -c conda-forge pygraphviz
```

Install CUDA 11.1 if needed
```python
conda install -c nvidia cudatoolkit=11.1
```

To install SourceCodeTools library run
```bash
git clone https://github.com/VitalyRomanov/method-embedding.git
cd method-embedding
pip install -e .
# pip install -e .[gpu]
```

### Installing Sourcetrail
Download a release from [Github repo](https://github.com/CoatiSoftware/Sourcetrail/releases) (latest tested version is 2020.1.117). Add Sourcetrail location to `PATH`
```bash
echo 'export PATH=/path/to/Sourcetrail_2020_1_117:$PATH' >> ~/.bashrc
```
Scripts that use Sourcetrail work on Linux, some issues were spotted on Macs. 

### Create Dataset
```bash
docker run -it -v "/full/path/to/data/folder":/dataset mortiv16/sourcetrail_indexer
```

[comment]: <> (```bash)

[comment]: <> (cd path/to/source_code)

[comment]: <> (echo "example\nexample2" | zsh -i scripts/data_collection/process_folders.sh)

[comment]: <> (bash -i scripts/data_extraction/process_sourcetrail.sh path/to/source_code)

[comment]: <> (python SourceCodeTools/code/data/sourcetrail/DatasetCreator2.py --bpe_tokenizer sentencepiece_bpe.model --track_offsets --do_extraction path/to/source_code path/to/graph)

[comment]: <> (```)