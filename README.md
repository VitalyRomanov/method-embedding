# Source Code Tools
Library for analyzing source code with graphs and NLP. What this repository can do:

1. Fetch source codes for packages in pip
2. Create indexes of python packages using [Sourcetrail](https://www.sourcetrail.com)
3. Convert Sourcetrail indexes into a connected graph
4. Build graphs for source codes from AST
5. Train Graph Neural Network for learning representations for source code
6. Predict Python types using NLP and graph embeddings

### Installation

```bash
git clone https://github.com/VitalyRomanov/method-embedding.git
cd method-embedding
pip install -e .
```