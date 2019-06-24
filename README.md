# Diffusion code for Landmark

This repository is re-implementation of [ducha-aiki/manifold-diffusion](https://github.com/ducha-aiki/manifold-diffusion) for [Google Landmark Retrieval 2019](https://www.kaggle.com/c/landmark-retrieval-2019/discussion) competition.

This program performs reranking index samples for each test sample in `submission.csv` .

## Requirements

- Ubuntu 16.04
- Python >= 3.7
- Conda 4.6.1

You also needs to prepare the input directory as below.

```
/path/to/ids-and-features/
|-- test_features.npy       # np.ndarray of test features. The shape is (num-samples, num-features)
|-- index_features.npy      # np.ndarray of index features. The shape is (num-samples, num-features) 
|-- test_ids.pickle         # pd.Series of test ids. 
`-- index_ids.pickle        # pd.Series of index ids. 

/path/to/similarity
`-- submission.csv         # pre-caluculated submission file. this file is reranking target.
```

## Example Usage

### Install dependencies

```bash
conda install -c pytorch faiss-gpu
pip isntall -r requirements.txt
```


### Calculate similarity before diffusion

```bash
export PYTHONPATH=.

# CPU-only 
python3 ./landmark_diffusion/pre_diffusion.py \
  -i /path/to/ids-and-features/ \
  -o /path/to/similarity/ \
  --nogpu \
  --norm \
  --K 50 \
  --QUERYKNN 10

# Use GPU
python3 ./landmark_diffusion/pre_diffusion.py \
  -i /path/to/ids-and-features/ \
  -o /path/to/similarity/ \
  --gpu \
  --norm \
  --K 50 \
  --QUERYKNN 10
```
  

### Generate submission.csv with diffusion

```bash
python3 ./landmark_diffusion/run_diffusion_with_similarity_for_subset.py \
  -i /path/to/ids-and-features/ \
  -s /path/to/similarity/ \
  -o /path/to/result/submission.csv \    # Result file of this program
  -R 10
```

