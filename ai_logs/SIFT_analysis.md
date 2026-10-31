# AI Log: complete creation of `SIFT_analysis.ipynb`

This log reconstructs the notebook from an empty file to its final executed state. Each cell has a implementation prompt followed by a `Response` containing the complete source for that cell. Outputs are included when the notebook captured them.

## Creation sequence

- Notebook cells: 8
- Markdown cells: 3
- Code cells: 5
- Execution order: top to bottom; each response depends on prior cells.

## Cell 1 > markdown

### Prompt

Create notebook cell 1 of 8 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
# Analysing the internals of SIFT 1M dataset

To understand how to benchmark the indexes on this datset we must first understand the dataset itself. The SIFT 1M dataset is a collection of 1 million SIFT descriptors, which are 128-dimensional vectors representing local features in images. These descriptors are widely used in computer vision tasks such as image matching and object recognition.
~~~~

## Cell 2 > code

### Prompt

Implement notebook code cell 2 of 8. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This script is to use pandas to read and anaylze the sift-128-euclidean.hdf5 file. The file contains a dataset of SIFT descriptors, which are 128-dimensional feature vectors used in computer vision tasks. The script reads the data from the HDF5 file and converts it into a pandas DataFrame for further analysis.
"""

import h5py
import pandas as pd

FILE_PATH = './data/raw/sift-128-euclidean.hdf5'


with h5py.File(FILE_PATH, 'r') as f:
    # Read the data from the HDF5 file
    train_data = f['train'][:]
    test_data = f['test'][:]
    neighbors_data = f['neighbors'][:]
    distances_data = f['distances'][:]

# Convert the data to pandas DataFrames
train_df = pd.DataFrame(train_data)
test_df = pd.DataFrame(test_data)
neighbors_df = pd.DataFrame(neighbors_data)
distances_df = pd.DataFrame(distances_data)

# Perform your analysis here
print("Train data shape:", train_df.shape)
print("Test data shape:", test_df.shape)
print("Neighbors data shape:", neighbors_df.shape)
print("Distances data shape:", distances_df.shape)
~~~~

### Captured output

~~~~text
Train data shape: (1000000, 128)
Test data shape: (10000, 128)
Neighbors data shape: (10000, 100)
Distances data shape: (10000, 100)
~~~~

Execution count recorded by the notebook: `2`.

## Cell 3 > code

### Prompt

Implement notebook code cell 3 of 8. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell will deal with checking the data inside the hdf5 file and analyzing it using pandas. The data is stored in a hierarchical structure, and we can access the datasets using their keys. We will read the datasets into pandas DataFrames for easier analysis and manipulation. 
"""

data = {
    'train': train_df,
    'test': test_df,
    'neighbors': neighbors_df,
    'distances': distances_df
}

# show first 5 rows of each dataset
for key, df in data.items():
    print(f"\n{key} dataset:")
    print(df.head())
~~~~

### Captured output

~~~~text

train dataset:
    0     1     2     3     4     5     6     7     8      9    ...    118  \
0   0.0  16.0  35.0   5.0  32.0  31.0  14.0  10.0  11.0   78.0  ...   23.0   
1  14.0  35.0  19.0  20.0   3.0   1.0  13.0  11.0  16.0  119.0  ...  119.0   
2   0.0   1.0   5.0   3.0  44.0  40.0  20.0  14.0  10.0  100.0  ...   34.0   
3  12.0  47.0  14.0  25.0   2.0   3.0   4.0   7.0  14.0  122.0  ...  122.0   
4   1.0   1.0   0.0   0.0  14.0  16.0  30.0  50.0   2.0   40.0  ...   70.0   

    119   120  121  122   123   124   125   126   127  
0  41.0   0.0  0.0  2.0   8.0  19.0  25.0  23.0   1.0  
1  52.0  15.0  2.0  0.0   0.0   0.0  11.0  21.0  33.0  
2  31.0   0.0  1.0  6.0  10.0  12.0   4.0  23.0  10.0  
3  43.0  15.0  1.0  0.0   0.0   0.0  27.0  29.0  21.0  
4  48.0   0.0  0.0  2.0   0.0   1.0   4.0  28.0  34.0  

[5 rows x 128 columns]

test dataset:
    0     1     2      3     4     5    6     7     8     9    ...   118  \
0   1.0   3.0  11.0  110.0  62.0  22.0  4.0   0.0  43.0  21.0  ...   2.0   
1  40.0  25.0  11.0    0.0  22.0  31.0  6.0   8.0  10.0   3.0  ...  60.0   
2  28.0   4.0   3.0    6.0   7.0   2.0  2.0  18.0  19.0  37.0  ...  59.0   
3  24.0  12.0  14.0    8.0   3.0  12.0  4.0   8.0   8.0  12.0  ...   8.0   
4   0.0   4.0  47.0   20.0   9.0   2.0  1.0   0.0  41.0  42.0  ...  21.0   

    119   120   121   122   123   124    125   126   127  
0   2.0  25.0  18.0   8.0   2.0  19.0   42.0  48.0  11.0  
1  40.0   4.0  30.0  23.0  32.0  10.0    3.0  19.0  13.0  
2  17.0   0.0   0.0   0.0   0.0   0.0    2.0  54.0  47.0  
3   0.0   1.0  39.0  15.0   0.0  14.0  103.0  16.0   0.0  
4   6.0   0.0   0.0   0.0   0.0   8.0   69.0  45.0   2.0  

[5 rows x 128 columns]

neighbors dataset:
       0       1       2       3       4       5       6       7       8   \
0  932085  934876  561813  708177  706771  695756  435345  701258  455537   
1  413247  413071  706838  880592  249062  400194  942339  880462  987636   
2  669835  408764  408462  408855  551661  861882  406273  406324  551743   
3  970797  125539   48044  191115  889039  546470  123985  546496  207768   
4  340871  748397  748193  175336  716433  112586  107281  416604  371420   

       9   ...      90      91      92      93      94      95      96  \
0  872728  ...  678385  158759  240996  931948   16429   91348   63349   
1  941776  ...  496660  947005  502126  888719  705714  125095  862007   
2  861530  ...  577672  331338  770558  516925  762311  468824  823033   
3  831905  ...  855019  564021  190979  807831  847143  885894  190642   
4  368246  ...  306702   34537  441587  242102  610628   31961  313434   

       97      98      99  
0  398306  931721  989762  
1  855176  846198  987074  
2  310475  971815  937903  
3  443325  886122  780591  
4  109870  693394  240166  

[5 rows x 100 columns]

distances dataset:
           0           1           2           3           4           5   \
0  232.871216  234.714722  243.989761  255.460373  256.314270  258.862885   
1  226.245438  229.281479  234.121765  234.493073  235.766830  235.785919   
2  175.476501  176.326401  183.324844  184.672684  186.273987  188.955017   
3  174.671112  175.678680  181.044189  186.751175  186.839508  189.034393   
4  214.275528  217.639618  219.433365  219.542709  229.257492  232.335968   

           6           7           8           9   ...          90  \
0  261.241272  264.280151  267.284485  268.069031  ...  288.234283   
1  237.147629  237.478424  238.296036  240.095810  ...  256.900757   
2  189.863113  190.328659  190.591187  190.651520  ...  213.103256   
3  190.018417  190.034210  190.339172  191.373978  ...  210.459015   
4  233.758423  236.249863  238.566559  238.843048  ...  258.065887   

           91          92          93          94          95          96  \
0  288.402496  288.473572  288.494354  288.629517  288.639923  288.688416   
1  257.015564  257.077820  257.093353  257.206146  257.408234  257.738251   
2  213.178329  213.307297  213.393997  213.525177  213.850418  213.897171   
3  210.637131  210.698837  210.907562  210.912308  210.931274  211.440292   
4  258.081390  258.247955  258.286652  258.304077  258.433746  258.629089   

           97          98          99  
0  288.750763  288.816559  288.904816  
1  257.827454  258.019379  258.226654  
2  213.920547  214.074753  214.107452  
3  211.442657  211.636475  211.782913  
4  258.897675  259.187195  259.273987  

[5 rows x 100 columns]
~~~~

Execution count recorded by the notebook: `3`.

## Cell 4 > markdown

### Prompt

Create notebook cell 4 of 8 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
### Understanding the overall structure of the dataset

The dataset is organized into four main components:
1. **Train data**: This contains the training set of SIFT descriptors.
2. **Test data**: This contains the test set of SIFT descriptors.
3. **Neighbors data**: This contains the indices of the nearest neighbors for each test sample.
4. **Distances data**: This contains the distances to the nearest neighbors for each test sample.

Based on the dataset structure, we can analyze the data to understand its characteristics and how it can be used for benchmarking.
~~~~

## Cell 5 > code

### Prompt

Implement notebook code cell 5 of 8. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell will be a random indexing implementation. For every query in the test dataset, we will randomly select a subset of the training dataset and compute the distances to find the nearest neighbors. This will help us understand the distribution of distances and the effectiveness of random indexing for approximate nearest neighbor search.
"""

# first we define a function to compute the Euclidean distance between two vectors
import numpy as np
import tqdm
import time

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# now we will randomly select a subset of 1000 samples from the training dataset for first 10 query in the test dataset, and return top 10 nearest neighbors for each query
# also calculate time of execution for each query and store it in a list
random_execution_times = []
random_nearest_neighbors_indices_list = list(list())
random_nearest_neighbors_distances_list = list(list())

for i in tqdm.tqdm(range(10)):
    query = test_df.iloc[i].values
    random_indices = np.random.choice(train_df.index, size=1000, replace=False)
    random_subset = train_df.iloc[random_indices].values
    # iloc is used to select rows by index, and values is used to get the underlying numpy array
    
    start_time = time.time()
    distances = np.array([euclidean_distance(query, train_sample) for train_sample in random_subset])
    random_nearest_neighbors_indices = np.argsort(distances)[:10]
    # also return the distances of the nearest neighbors
    random_nearest_neighbors_distances = distances[random_nearest_neighbors_indices]
    random_execution_time = time.time() - start_time
    random_execution_times.append(random_execution_time)
    
    print(f"Query {i}: Nearest neighbors indices: {random_nearest_neighbors_indices}, Execution time: {random_execution_time:.4f} seconds")
    print(f"Query {i}: Nearest neighbors distances: {random_nearest_neighbors_distances}")
    random_nearest_neighbors_indices_list.append(random_nearest_neighbors_indices)
    random_nearest_neighbors_distances_list.append(random_nearest_neighbors_distances)
print(random_nearest_neighbors_indices_list)
print(random_nearest_neighbors_distances_list)
~~~~

### Captured output

~~~~text
  0%|          | 0/10 [00:00<>, >it/s]
 40%|████      | 4/10 [00:00<00:00, 39.48it/s]
Query 0: Nearest neighbors indices: [811 144 880 488 635 107 554 527 989  17], Execution time: 0.0033 seconds
Query 0: Nearest neighbors distances: [293.81116 322.8746  332.47104 335.70374 336.81003 342.39014 343.45886
 343.4647  344.39804 347.31253]
Query 1: Nearest neighbors indices: [599 919 808 121 358 817 334 966 856 739], Execution time: 0.0020 seconds
Query 1: Nearest neighbors distances: [266.26114 280.65994 281.09073 292.30804 300.90695 302.21848 302.36237
 305.8104  307.12213 307.6378 ]
Query 2: Nearest neighbors indices: [995 983 113 846 644 978 634 130 105 885], Execution time: 0.0019 seconds
Query 2: Nearest neighbors distances: [251.1394  254.58202 260.48032 261.69638 261.811   268.71545 278.70056
 284.3941  284.6682  288.8356 ]
Query 3: Nearest neighbors indices: [418 326 126 864 172 673 671 237 959 784], Execution time: 0.0017 seconds
Query 3: Nearest neighbors distances: [212.71812 225.61914 231.79733 251.86107 259.69214 278.41516 283.3849
 283.6424  300.68256 301.46973]
Query 4: Nearest neighbors indices: [121 314 602  97 379 605 745 217 627 338], Execution time: 0.0018 seconds
Query 4: Nearest neighbors distances: [307.61502 321.6862  323.31253 324.5905  326.97552 327.95578 329.1322
 331.62027 333.59857 335.18652]
Query 5: Nearest neighbors indices: [791 626 943 163 421 690 244 715 497 426], Execution time: 0.0018 seconds
Query 5: Nearest neighbors distances: [304.08057 325.69772 335.6844  340.39536 342.26743 349.98285 356.1727
 359.01254 359.92776 361.2326 ]

 90%|█████████ | 9/10 [00:00<00:00, 41.46it/s]
Query 6: Nearest neighbors indices: [801 353 103  34 153 664 732 266 542 339], Execution time: 0.0018 seconds
Query 6: Nearest neighbors distances: [290.85046 294.4724  296.5485  305.45538 313.70688 315.      317.47913
 320.94702 322.6763  322.93808]
Query 7: Nearest neighbors indices: [921 777 580 780   6  97 627  29 208 778], Execution time: 0.0018 seconds
Query 7: Nearest neighbors distances: [258.14145 266.85016 269.4086  277.67065 289.63254 298.46606 300.24823
 301.894   301.93542 303.0858 ]
Query 8: Nearest neighbors indices: [925 538   2 625 247 962 954  69 650 897], Execution time: 0.0018 seconds
Query 8: Nearest neighbors distances: [268.6522  277.14438 279.5568  298.34375 298.7541  299.58136 304.86063
 305.10654 306.77026 307.14816]

100%|██████████| 10/10 [00:00<00:00, 40.81it/s]
Query 9: Nearest neighbors indices: [828 958 946 507 442 188 604 696 933 473], Execution time: 0.0017 seconds
Query 9: Nearest neighbors distances: [288.8858  293.5183  297.64407 298.20294 303.9671  304.0592  305.21796
 307.41827 308.98544 319.3869 ]
[array([811, 144, 880, 488, 635, 107, 554, 527, 989,  17]), array([599, 919, 808, 121, 358, 817, 334, 966, 856, 739]), array([995, 983, 113, 846, 644, 978, 634, 130, 105, 885]), array([418, 326, 126, 864, 172, 673, 671, 237, 959, 784]), array([121, 314, 602,  97, 379, 605, 745, 217, 627, 338]), array([791, 626, 943, 163, 421, 690, 244, 715, 497, 426]), array([801, 353, 103,  34, 153, 664, 732, 266, 542, 339]), array([921, 777, 580, 780,   6,  97, 627,  29, 208, 778]), array([925, 538,   2, 625, 247, 962, 954,  69, 650, 897]), array([828, 958, 946, 507, 442, 188, 604, 696, 933, 473])]
[array([293.81116, 322.8746 , 332.47104, 335.70374, 336.81003, 342.39014,
       343.45886, 343.4647 , 344.39804, 347.31253], dtype=float32), array([266.26114, 280.65994, 281.09073, 292.30804, 300.90695, 302.21848,
       302.36237, 305.8104 , 307.12213, 307.6378 ], dtype=float32), array([251.1394 , 254.58202, 260.48032, 261.69638, 261.811  , 268.71545,
       278.70056, 284.3941 , 284.6682 , 288.8356 ], dtype=float32), array([212.71812, 225.61914, 231.79733, 251.86107, 259.69214, 278.41516,
       283.3849 , 283.6424 , 300.68256, 301.46973], dtype=float32), array([307.61502, 321.6862 , 323.31253, 324.5905 , 326.97552, 327.95578,
       329.1322 , 331.62027, 333.59857, 335.18652], dtype=float32), array([304.08057, 325.69772, 335.6844 , 340.39536, 342.26743, 349.98285,
       356.1727 , 359.01254, 359.92776, 361.2326 ], dtype=float32), array([290.85046, 294.4724 , 296.5485 , 305.45538, 313.70688, 315.     ,
       317.47913, 320.94702, 322.6763 , 322.93808], dtype=float32), array([258.14145, 266.85016, 269.4086 , 277.67065, 289.63254, 298.46606,
       300.24823, 301.894  , 301.93542, 303.0858 ], dtype=float32), array([268.6522 , 277.14438, 279.5568 , 298.34375, 298.7541 , 299.58136,
       304.86063, 305.10654, 306.77026, 307.14816], dtype=float32), array([288.8858 , 293.5183 , 297.64407, 298.20294, 303.9671 , 304.0592 ,
       305.21796, 307.41827, 308.98544, 319.3869 ], dtype=float32)]
~~~~

Execution count recorded by the notebook: `4`.

## Cell 6 > code

### Prompt

Implement notebook code cell 6 of 8. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell will be a brute force implementation. For every query in the test dataset, we will compute the distances to all samples in the training dataset and find the nearest neighbors. This will help us understand the distribution of distances and the effectiveness of brute force search for exact nearest neighbor search. This is similar to KNN search, but we will not use any libraries for this implementation. We will implement the distance calculation and nearest neighbor search from scratch.
"""

exact_execution_times = []
exact_nearest_neighbors_indices_list = list(list())
exact_nearest_neighbors_distances_list = list(list())

for i in tqdm.tqdm(range(10)):
    query = test_df.iloc[i].values
    start_time = time.time()
    distances = np.array([euclidean_distance(query, train_sample) for train_sample in train_df.values])
    exact_nearest_neighbors_indices = np.argsort(distances)[:10]
    exact_nearest_neighbors_distances = distances[exact_nearest_neighbors_indices]
    exact_execution_time = time.time() - start_time
    exact_execution_times.append(exact_execution_time)
    
    print(f"Query {i}: Nearest neighbors indices: {exact_nearest_neighbors_indices}, Execution time: {exact_execution_time:.4f} seconds")
    print(f"Query {i}: Nearest neighbors distances: {exact_nearest_neighbors_distances}")
    exact_nearest_neighbors_indices_list.append(exact_nearest_neighbors_indices)
    exact_nearest_neighbors_distances_list.append(exact_nearest_neighbors_distances)
~~~~

### Captured output

~~~~text
 10%|█         | 1/10 [00:01<00:15,  1.73s/it]
Query 0: Nearest neighbors indices: [932085 934876 561813 708177 706771 695756 435345 701258 455537 872728], Execution time: 1.7286 seconds
Query 0: Nearest neighbors distances: [232.87122 234.71472 243.98976 255.46037 256.31427 258.86288 261.24127
 264.28015 267.2845  268.06903]

 20%|██        | 2/10 [00:03<00:13,  1.71s/it]
Query 1: Nearest neighbors indices: [413247 413071 706838 880592 249062 400194 942339 880462 987636 941776], Execution time: 1.6931 seconds
Query 1: Nearest neighbors distances: [226.24544 229.28148 234.12177 234.49307 235.76683 235.78592 237.14763
 237.47842 238.29604 240.09581]

 30%|███       | 3/10 [00:05<00:11,  1.70s/it]
Query 2: Nearest neighbors indices: [669835 408764 408462 408855 551661 861882 406273 406324 551743 861530], Execution time: 1.6949 seconds
Query 2: Nearest neighbors distances: [175.4765  176.3264  183.32484 184.67268 186.27399 188.95502 189.86311
 190.32866 190.59119 190.65152]

 40%|████      | 4/10 [00:06<00:10,  1.69s/it]
Query 3: Nearest neighbors indices: [970797 125539  48044 191115 889039 546470 123985 546496 207768 831905], Execution time: 1.6750 seconds
Query 3: Nearest neighbors distances: [174.67111 175.67868 181.04419 186.75117 186.83951 189.0344  190.01842
 190.03421 190.33917 191.37398]

 50%|█████     | 5/10 [00:08<00:08,  1.69s/it]
Query 4: Nearest neighbors indices: [340871 748397 748193 175336 716433 112586 107281 416604 371420 368246], Execution time: 1.6819 seconds
Query 4: Nearest neighbors distances: [214.27553 217.63962 219.43336 219.54271 229.25749 232.33597 233.75842
 236.24986 238.56656 238.84305]

 60%|██████    | 6/10 [00:10<00:06,  1.67s/it]
Query 5: Nearest neighbors indices: [187470  67875 220473 460733 896005  27746 943689 669622 947299 515025], Execution time: 1.6400 seconds
Query 5: Nearest neighbors distances: [253.8858  256.4586  256.4586  258.05618 265.76492 266.06015 267.84885
 268.38593 270.49585 270.9631 ]

 70%|███████   | 7/10 [00:11<00:04,  1.67s/it]
Query 6: Nearest neighbors indices: [402219 982379 652078 880346  59540 511412 595673 703786 982409 596714], Execution time: 1.6509 seconds
Query 6: Nearest neighbors distances: [224.32788 226.55463 230.85927 231.01732 236.1906  236.37682 236.57768
 238.5917  238.73207 238.83467]

 80%|████████  | 8/10 [00:13<00:03,  1.66s/it]
Query 7: Nearest neighbors indices: [906750 618842 807599 569178 107468 207868 476215 762217 762237   5447], Execution time: 1.6392 seconds
Query 7: Nearest neighbors distances: [154.87415 161.33815 163.5237  164.35023 164.44148 166.003   171.31258
 172.83807 173.40128 173.94827]

 90%|█████████ | 9/10 [00:15<00:01,  1.66s/it]
Query 8: Nearest neighbors indices: [376328 325865 323160 376277 323464 724549 176970 695175 803977 181435], Execution time: 1.6532 seconds
Query 8: Nearest neighbors distances: [213.66095 224.81326 225.56818 230.56453 231.48434 233.76056 235.65228
 236.17366 236.65376 236.94302]

100%|██████████| 10/10 [00:16<00:00,  1.68s/it]
Query 9: Nearest neighbors indices: [178811 177646 181997 181605 821938 807785 716433 358802 803100 866737], Execution time: 1.6963 seconds
Query 9: Nearest neighbors distances: [210.15947 217.01613 218.54062 219.29889 221.7228  226.71347 227.31476
 230.73145 230.91124 231.64412]
~~~~

Execution count recorded by the notebook: `5`.

## Cell 7 > code

### Prompt

Implement notebook code cell 7 of 8. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
K = 10
N_QUERIES = 10

# Ground truth (slice rows AND columns in one step)
true_idx = neighbors_df.iloc[:N_QUERIES, :K].values
true_dist = np.round(distances_df.iloc[:N_QUERIES, :K].values, 2)

def compare(name, idx_list, dist_list):
    idx = np.array(idx_list)[:N_QUERIES, :K]
    dist = np.round(np.array(dist_list)[:N_QUERIES, :K], 2)

    # Per-query results
    index_match = (idx == true_idx).astype(int)               # (N_QUERIES, K) 1 = same position
    dist_diff = np.abs(dist - true_dist).mean(axis=1)         # mean abs distance error per query

    for i in range(N_QUERIES):
        print(f"Query {i} | {name} vs true | index matches: {index_match[i].tolist()}")
        print(f"Query {i} | {name} vs true | mean abs distance diff: {dist_diff[i]:.4f}")

    return index_match, dist_diff

rand_idx_match, rand_dist_diff = compare(
    "random", random_nearest_neighbors_indices_list, random_nearest_neighbors_distances_list)
exact_idx_match, exact_dist_diff = compare(
    "exact", exact_nearest_neighbors_indices_list, exact_nearest_neighbors_distances_list)

# Store in a DataFrame for further analysis
results_df = pd.DataFrame({
    "query": range(N_QUERIES),
    "random_index_match_rate": rand_idx_match.mean(axis=1),
    "random_mean_abs_dist_diff": rand_dist_diff,
    "exact_index_match_rate": exact_idx_match.mean(axis=1),
    "exact_mean_abs_dist_diff": exact_dist_diff,
})
print(results_df)
~~~~

### Captured output

~~~~text
Query 0 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 0 | random vs true | mean abs distance diff: 79.9610
Query 1 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 1 | random vs true | mean abs distance diff: 59.7650
Query 2 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 2 | random vs true | mean abs distance diff: 83.8570
Query 3 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 3 | random vs true | mean abs distance diff: 77.3510
Query 4 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 4 | random vs true | mean abs distance diff: 98.1780
Query 5 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 5 | random vs true | mean abs distance diff: 80.0060
Query 6 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 6 | random vs true | mean abs distance diff: 76.2030
Query 7 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 7 | random vs true | mean abs distance diff: 120.1320
Query 8 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 8 | random vs true | mean abs distance diff: 64.0660
Query 9 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 9 | random vs true | mean abs distance diff: 79.3260
Query 0 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 0 | exact vs true | mean abs distance diff: 0.0000
Query 1 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 1 | exact vs true | mean abs distance diff: 0.0000
Query 2 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 2 | exact vs true | mean abs distance diff: 0.0000
Query 3 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 3 | exact vs true | mean abs distance diff: 0.0000
Query 4 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 4 | exact vs true | mean abs distance diff: 0.0000
Query 5 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 5 | exact vs true | mean abs distance diff: 0.0000
Query 6 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 6 | exact vs true | mean abs distance diff: 0.0000
Query 7 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 7 | exact vs true | mean abs distance diff: 0.0000
Query 8 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 8 | exact vs true | mean abs distance diff: 0.0000
Query 9 | exact vs true | index matches: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
Query 9 | exact vs true | mean abs distance diff: 0.0000
   query  random_index_match_rate  random_mean_abs_dist_diff  \
0      0                      0.0                  79.960999   
1      1                      0.0                  59.765003   
2      2                      0.0                  83.857010   
3      3                      0.0                  77.350998   
4      4                      0.0                  98.178001   
5      5                      0.0                  80.005997   
6      6                      0.0                  76.202995   
7      7                      0.0                 120.132004   
8      8                      0.0                  64.065994   
9      9                      0.0                  79.326004   

   exact_index_match_rate  exact_mean_abs_dist_diff  
0                     1.0                       0.0  
1                     1.0                       0.0  
2                     1.0                       0.0  
3                     1.0                       0.0  
4                     1.0                       0.0  
5                     1.0                       0.0  
6                     1.0                       0.0  
7                     1.0                       0.0  
8                     1.0                       0.0  
9                     1.0                       0.0
~~~~

Execution count recorded by the notebook: `7`.

## Cell 8 > markdown

### Prompt

Create notebook cell 8 of 8 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Conclusion

The analysis shows that the difference in processing time between random and exact nearest neighbor search is significant, with exact search taking considerably longer. This highlights the importance of using efficient indexing methods for large datasets like SIFT 1M. By understanding the dataset's structure and characteristics, we can better design and evaluate indexing algorithms to improve performance in real-world applications. Random search is fast but not accurate, while exact search is accurate but slow, the difference in time is of the order of several seconds per query.

This is where we need ANN (Approximate Nearest Neighbor) search methods, which provide a trade-off between accuracy and speed. ANN algorithms can significantly reduce the time required to find nearest neighbors while maintaining acceptable levels of accuracy, making them suitable for large-scale datasets like SIFT 1M.
~~~~
