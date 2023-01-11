# Benchmarking existing ANN Indexes

<!-- ABOUT THE PROJECT -->
## About The Project

Run benchmarks for approximate nearest neighbor algorithms search on your dataset.

<!-- GETTING STARTED -->
## Getting Started

In order to run Benchmarks you need to provide np.array with embeddings(in `.npy` file) and specify experiments parameters(dataset_path, search parameters, and list indexes with parameters to test) in a config file.
You can find a sample config file in `configs/test_run.yaml`.

## Installation

1. Clone the repo

   ```bash
   git clone https://github.com/your_username_/Project-Name.git
   ```

2. Install requirements

   ```bash
   pip install -r requirements.txt
   ```

<!-- USAGE EXAMPLES -->
## Usage

```bash
python run.py --config_pth configs/test_run.yaml  # Path to your config file with experiments parameters
```

## Implemented

* [Annoy](https://github.com/spotify/annoy)
* [scikit-learn](http://scikit-learn.org/stable/modules/neighbors.html)
* [hnswlib](https://github.com/nmslib/hnsw)
* [FAISS](https://github.com/facebookresearch/faiss.git)
* [ScaNN](https://github.com/google-research/google-research/tree/master/scann)

<!-- ROADMAP -->
## Roadmap

* [x] Add Readme

* [ ] Add requirements.txt

* [x] Add ScaNN

* [ ] Add run results

* [x] Add About The Project

* [ ] Cleanup
