# AI Log: complete creation of `glove_analysis.ipynb`

This log reconstructs the notebook from an empty file to its final executed state. Each cell has a implementation prompt followed by a `Response` containing the complete source for that cell. Outputs are included when the notebook captured them.

## Creation sequence

- Notebook cells: 7
- Markdown cells: 2
- Code cells: 5
- Execution order: top to bottom; each response depends on prior cells.

## Cell 1 > markdown

### Prompt

Create notebook cell 1 of 7 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
# Analysing the Internals of Glove 100 angular dataset

The Glove 100 angular dataset is a collection of word embeddings that represent words in a high-dimensional space. Each word is represented as a vector of 100 dimensions, capturing semantic relationships between words based on their co-occurrence in large text corpora.
~~~~

## Cell 2 > code

### Prompt

Implement notebook code cell 2 of 7. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This script is to use pandas to read and anaylze the sift-128-euclidean.hdf5 file. The file contains a dataset of SIFT descriptors, which are 128-dimensional feature vectors used in computer vision tasks. The script reads the data from the HDF5 file and converts it into a pandas DataFrame for further analysis.
"""

import h5py
import pandas as pd

FILE_PATH = './data/raw/glove-100-angular.hdf5'


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
Train data shape: (1183514, 100)
Test data shape: (10000, 100)
Neighbors data shape: (10000, 100)
Distances data shape: (10000, 100)
~~~~

Execution count recorded by the notebook: `1`.

## Cell 3 > code

### Prompt

Implement notebook code cell 3 of 7. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

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
        0        1         2        3         4        5         6        7   \
0 -0.11333  0.48402  0.090771 -0.22439  0.034206 -0.55831  0.041849 -0.53573   
1 -0.99544 -2.36510 -0.243320 -1.03210  0.420520 -1.18170 -0.164510 -1.68300   
2 -0.10185  0.59817  0.202810  0.25243 -1.058300 -0.33025  1.515000 -0.31013   
3 -0.73278 -0.47153  1.331200  0.94514  0.391930 -0.15517  0.770010 -0.48571   
4 -0.21327  0.63622 -0.503040  0.17258 -0.537730 -0.51566 -0.499850 -1.02340   

        8        9   ...       90       91       92        93       94  \
0  0.18809 -0.58722  ... -0.17456  0.53571 -0.16537  0.135820 -0.56016   
1  0.49673 -0.27258  ...  0.40869 -0.08503  0.64130 -0.840440 -0.74325   
2 -0.52481  0.63273  ... -0.17440 -1.38970  0.37120  1.538800 -0.04959   
3 -0.25486 -0.51172  ...  0.46976 -0.70778  0.35542  1.015900  0.38979   
4 -0.35922 -1.39940  ... -0.26148 -0.66992 -0.58094  0.040179  0.33297   

         95        96        97        98        99  
0  0.016964  0.127700  0.940710 -0.226080 -0.021106  
1 -0.194260  0.098722  0.326480 -0.676210 -0.626920  
2 -0.307800 -0.696260 -0.042742  0.472870  0.068169  
3  0.570780  0.194370  0.890640 -0.808210  0.921340  
4  0.404820  0.344660  0.681010 -0.038315  0.093102  

[5 rows x 100 columns]

test dataset:
        0        1        2        3        4        5        6         7   \
0  0.39553  0.23048  0.82722  0.10453 -0.69281 -0.83357 -0.49049 -0.036362   
1 -0.44396  0.71301 -0.50031 -0.25644 -0.55146  0.25119 -0.59982 -0.433980   
2 -0.10801 -0.48462 -0.43255 -0.35585  0.87992 -0.26895 -0.23992 -1.030000   
3 -0.49479 -0.63801 -0.42185  0.99306 -0.87891  0.73786 -0.38802  0.143300   
4 -0.20213 -1.53800  1.56530 -0.69704  0.10239 -1.45390  0.28788 -0.099137   

        8        9   ...        90        91       92       93        94  \
0 -0.48396 -0.44315  ...  0.150760 -0.010834 -0.11687  0.54617 -0.364120   
1 -0.55832 -0.82649  ...  0.622610 -0.068024  0.47316 -0.27670  0.021853   
2 -0.86838  0.62238  ... -1.063000 -0.192700  0.26283  0.38861  0.015967   
3  0.20837  0.32576  ... -0.041551  0.489450  0.93903  0.21480  0.152830   
4 -0.31745  0.46771  ...  0.348270  0.320160  1.02740 -0.67064  0.432490   

        95        96       97       98       99  
0  0.50771  0.682840  0.79829 -0.29665 -0.14753  
1 -0.52443  0.152540  0.18179 -0.79941  0.89957  
2 -0.40039  0.065531  0.37788 -1.09820  0.18677  
3 -1.15500  0.126150  0.97227 -0.59029  1.22610  
4 -0.15127  0.320760 -0.40570 -0.70136  1.43500  

[5 rows x 100 columns]

neighbors dataset:
        0       1       2       3        4       5       6        7        8   \
0    97478  262700  846101  671078   232287  727732  544474  1133489   723915   
1   875925  903728  144313  507188   547349  643514  663762   675600   891287   
2  1046944  809599  531832  323299  1077727  704557   62703   988527   377259   
3  1161354  691655  326662  276040   760347  444556  969229  1013230   820889   
4   764146  964066   78273  831701   404970  132359  988497   733344  1058389   

       9   ...       90      91       92       93       94       95      96  \
0  660281  ...    74880  418892   687317  1008844  1011545   983601  340497   
1  712921  ...  1035617  232836    85305   726292  1052632   689129  109186   
2  625676  ...  1003460  176037   994300   667108   501274  1066386  732333   
3  377815  ...   272678  971913   111403  1039443   534321   733979  378715   
4  558150  ...   552645  166075  1051472   352156  1108908   364948  462379   

       97       98      99  
0  598329   944409  725625  
1  857806   543125  973672  
2  821775   340919  685590  
3  562767  1119971  201781  
4  961106    15857  566217  

[5 rows x 100 columns]

distances dataset:
         0         1         2         3         4         5         6   \
0  0.430394  0.432385  0.433091  0.433414  0.437589  0.439799  0.449643   
1  0.358507  0.368545  0.386401  0.386546  0.387637  0.393225  0.393659   
2  0.362573  0.386137  0.395487  0.397532  0.405297  0.410762  0.417822   
3  0.432271  0.450587  0.450986  0.462078  0.464149  0.465605  0.466690   
4  0.413611  0.416309  0.432118  0.435037  0.437729  0.440159  0.440813   

         7         8         9   ...        90        91        92        93  \
0  0.450096  0.451968  0.452228  ...  0.486494  0.486943  0.486994  0.487162   
1  0.396342  0.396816  0.398280  ...  0.437824  0.438909  0.439102  0.439192   
2  0.420367  0.423940  0.425181  ...  0.481994  0.482022  0.482088  0.482194   
3  0.469045  0.474185  0.477329  ...  0.517717  0.518037  0.518438  0.518936   
4  0.441799  0.446078  0.447504  ...  0.510246  0.510292  0.510299  0.510408   

         94        95        96        97        98        99  
0  0.487590  0.487646  0.487866  0.488100  0.488175  0.488259  
1  0.439230  0.439732  0.440000  0.440210  0.440406  0.440483  
2  0.482349  0.482561  0.483771  0.484237  0.484541  0.484790  
3  0.519972  0.520265  0.520369  0.520441  0.520499  0.520674  
4  0.510464  0.510482  0.510909  0.511252  0.511325  0.511408  

[5 rows x 100 columns]
~~~~

Execution count recorded by the notebook: `2`.

## Cell 4 > markdown

### Prompt

Create notebook cell 4 of 7 as documentation. Preserve the assignment context, technical explanations, measurement rules, and headings exactly as needed for a coherent from-scratch ANN experiment.

### Response

~~~~markdown
## Understanding overall structure of the dataset

The dataset is organized in a hierarchical structure, with groups and datasets that store the word embeddings and associated metadata. The main components of the dataset include:
1. **Train data**: This contains the training set of word embeddings, which are used to learn the relationships between words.
2. **Test data**: This contains the test set of word embeddings, which are used to evaluate the performance of models trained on the training data.
3. **Neighbors data**: This contains the nearest neighbors for each word in the test set, which are used to evaluate the quality of the word embeddings.
4. **Distances data**: This contains the distances between each word in the test set and its nearest neighbors, which are used to evaluate the quality of the word embeddings.
~~~~

## Cell 5 > code

### Prompt

Implement notebook code cell 5 of 7. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

### Response

~~~~python
"""
This cell will be a random indexing implementation. For every query in the test dataset, we will randomly select a subset of the training dataset and compute the distances to find the nearest neighbors. This will help us understand the distribution of distances and the effectiveness of random indexing for approximate nearest neighbor search.
"""

# first we define a function to compute the Euclidean distance between two vectors
import numpy as np
import tqdm
import time

# for glove 100 angular dataset, we will use cosine distance instead of Euclidean distance
# formula for cosine distance is 1 - (A.B) / (||A|| * ||B||)
def cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

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
    distances = np.array([cosine_distance(query, x) for x in random_subset])
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
 60%|██████    | 6/10 [00:00<00:00, 26.07it/s]
Query 0: Nearest neighbors indices: [124 932 459 369 365 585 856 589 764 732], Execution time: 0.0040 seconds
Query 0: Nearest neighbors distances: [0.52932966 0.5297325  0.57760036 0.57803214 0.5905267  0.5917449
 0.59558374 0.6027187  0.60536367 0.6112678 ]
Query 1: Nearest neighbors indices: [416 871 229 712 384 265 145  33 479  64], Execution time: 0.0036 seconds
Query 1: Nearest neighbors distances: [0.41938198 0.5115577  0.5206841  0.54189867 0.54782104 0.5547854
 0.55986834 0.5758499  0.5815945  0.5834988 ]
Query 2: Nearest neighbors indices: [828 814 823 567 426 547 530 495 341  16], Execution time: 0.0037 seconds
Query 2: Nearest neighbors distances: [0.4251811  0.5665781  0.57106805 0.58154774 0.59638923 0.61348206
 0.6338906  0.64380133 0.644757   0.6532109 ]
Query 3: Nearest neighbors indices: [200 238  24 768 464 807  88 111 730  67], Execution time: 0.0039 seconds
Query 3: Nearest neighbors distances: [0.59247625 0.6011677  0.601867   0.6036888  0.6112268  0.61152434
 0.638812   0.6406209  0.64127696 0.64905745]
Query 4: Nearest neighbors indices: [886 418 922 765 719 329  33 452 485  39], Execution time: 0.0036 seconds
Query 4: Nearest neighbors distances: [0.5203297  0.5717723  0.5916706  0.6030659  0.60428524 0.60990226
 0.61475945 0.6228303  0.62567127 0.6291808 ]
Query 5: Nearest neighbors indices: [ 89 133 329 139 784 433 849 186 835 971], Execution time: 0.0036 seconds
Query 5: Nearest neighbors distances: [0.59284675 0.5966362  0.6247882  0.6420715  0.66870403 0.67302746
 0.6815827  0.68289286 0.6843567  0.6863388 ]
Query 6: Nearest neighbors indices: [472 166 132  15 956 892 679 527 347 781], Execution time: 0.0039 seconds
Query 6: Nearest neighbors distances: [0.5000465  0.55997205 0.66189045 0.67024994 0.69552135 0.7011429
 0.7042692  0.7101495  0.71111155 0.7192327 ]

100%|██████████| 10/10 [00:00<00:00, 27.99it/s]
Query 7: Nearest neighbors indices: [863 422 170 122 998 360 152 576 603 217], Execution time: 0.0039 seconds
Query 7: Nearest neighbors distances: [0.5841369  0.58828944 0.59520435 0.5962031  0.6131913  0.6140554
 0.619784   0.6251745  0.6294094  0.63930786]
Query 8: Nearest neighbors indices: [219  29 747 936 120 932 637 342 887 947], Execution time: 0.0039 seconds
Query 8: Nearest neighbors distances: [0.6353389  0.6438589  0.6646058  0.66918266 0.66927266 0.66984963
 0.67023253 0.6927533  0.69587636 0.7117095 ]
Query 9: Nearest neighbors indices: [919 950 960 364 939 699 288 904  33 516], Execution time: 0.0039 seconds
Query 9: Nearest neighbors distances: [0.5070495  0.55207187 0.5831878  0.59166324 0.615345   0.62024033
 0.6232034  0.62641025 0.6353526  0.6394695 ]
[array([124, 932, 459, 369, 365, 585, 856, 589, 764, 732]), array([416, 871, 229, 712, 384, 265, 145,  33, 479,  64]), array([828, 814, 823, 567, 426, 547, 530, 495, 341,  16]), array([200, 238,  24, 768, 464, 807,  88, 111, 730,  67]), array([886, 418, 922, 765, 719, 329,  33, 452, 485,  39]), array([ 89, 133, 329, 139, 784, 433, 849, 186, 835, 971]), array([472, 166, 132,  15, 956, 892, 679, 527, 347, 781]), array([863, 422, 170, 122, 998, 360, 152, 576, 603, 217]), array([219,  29, 747, 936, 120, 932, 637, 342, 887, 947]), array([919, 950, 960, 364, 939, 699, 288, 904,  33, 516])]
[array([0.52932966, 0.5297325 , 0.57760036, 0.57803214, 0.5905267 ,
       0.5917449 , 0.59558374, 0.6027187 , 0.60536367, 0.6112678 ],
      dtype=float32), array([0.41938198, 0.5115577 , 0.5206841 , 0.54189867, 0.54782104,
       0.5547854 , 0.55986834, 0.5758499 , 0.5815945 , 0.5834988 ],
      dtype=float32), array([0.4251811 , 0.5665781 , 0.57106805, 0.58154774, 0.59638923,
       0.61348206, 0.6338906 , 0.64380133, 0.644757  , 0.6532109 ],
      dtype=float32), array([0.59247625, 0.6011677 , 0.601867  , 0.6036888 , 0.6112268 ,
       0.61152434, 0.638812  , 0.6406209 , 0.64127696, 0.64905745],
      dtype=float32), array([0.5203297 , 0.5717723 , 0.5916706 , 0.6030659 , 0.60428524,
       0.60990226, 0.61475945, 0.6228303 , 0.62567127, 0.6291808 ],
      dtype=float32), array([0.59284675, 0.5966362 , 0.6247882 , 0.6420715 , 0.66870403,
       0.67302746, 0.6815827 , 0.68289286, 0.6843567 , 0.6863388 ],
      dtype=float32), array([0.5000465 , 0.55997205, 0.66189045, 0.67024994, 0.69552135,
       0.7011429 , 0.7042692 , 0.7101495 , 0.71111155, 0.7192327 ],
      dtype=float32), array([0.5841369 , 0.58828944, 0.59520435, 0.5962031 , 0.6131913 ,
       0.6140554 , 0.619784  , 0.6251745 , 0.6294094 , 0.63930786],
      dtype=float32), array([0.6353389 , 0.6438589 , 0.6646058 , 0.66918266, 0.66927266,
       0.66984963, 0.67023253, 0.6927533 , 0.69587636, 0.7117095 ],
      dtype=float32), array([0.5070495 , 0.55207187, 0.5831878 , 0.59166324, 0.615345  ,
       0.62024033, 0.6232034 , 0.62641025, 0.6353526 , 0.6394695 ],
      dtype=float32)]
~~~~

## Cell 6 > code

### Prompt

Implement notebook code cell 6 of 7. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

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
    distances = np.array([cosine_distance(query, x) for x in train_df.values])
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
 10%|█         | 1/10 [00:04<00:39,  4.37s/it]
Query 0: Nearest neighbors indices: [  97478  262700  846101  671078  232287  727732  544474 1133489  723915
  660281], Execution time: 4.3690 seconds
Query 0: Nearest neighbors distances: [0.4303941  0.4323855  0.43309093 0.43341446 0.4375894  0.4397995
 0.4496429  0.4500963  0.45196807 0.4522282 ]

 20%|██        | 2/10 [00:08<00:34,  4.35s/it]
Query 1: Nearest neighbors indices: [875925 903728 144313 507188 547349 643514 663762 675600 891287 712921], Execution time: 4.3284 seconds
Query 1: Nearest neighbors distances: [0.3585065  0.36854464 0.386401   0.38654637 0.3876366  0.3932249
 0.39365864 0.39634246 0.3968156  0.39827985]

 30%|███       | 3/10 [00:13<00:30,  4.35s/it]
Query 2: Nearest neighbors indices: [1046944  809599  531832  323299 1077727  704557   62703  988527  377259
  625676], Execution time: 4.3521 seconds
Query 2: Nearest neighbors distances: [0.36257327 0.3861372  0.39548653 0.39753222 0.4052965  0.41076195
 0.41782224 0.42036742 0.4239397  0.4251811 ]

 40%|████      | 4/10 [00:17<00:25,  4.33s/it]
Query 3: Nearest neighbors indices: [1161354  691655  326662  276040  760347  444556  969229 1013230  820889
  377815], Execution time: 4.3029 seconds
Query 3: Nearest neighbors distances: [0.43227148 0.45058674 0.45098633 0.46207786 0.46414912 0.46560508
 0.4666896  0.46904457 0.47418535 0.47732937]

 50%|█████     | 5/10 [00:21<00:21,  4.33s/it]
Query 4: Nearest neighbors indices: [ 764146  964066   78273  831701  404970  132359  988497  733344 1058389
  558150], Execution time: 4.3196 seconds
Query 4: Nearest neighbors distances: [0.41361058 0.41630918 0.43211782 0.4350369  0.43772864 0.44015896
 0.44081283 0.44179946 0.44607812 0.4475038 ]

 60%|██████    | 6/10 [00:26<00:17,  4.37s/it]
Query 5: Nearest neighbors indices: [  75098  820497  275087 1105740  245039  310142  590034   38015  462762
  617324], Execution time: 4.4572 seconds
Query 5: Nearest neighbors distances: [0.48219508 0.49535918 0.49674034 0.4992544  0.500753   0.5034882
 0.5081135  0.5103185  0.51346076 0.5148076 ]

 70%|███████   | 7/10 [00:30<00:13,  4.36s/it]
Query 6: Nearest neighbors indices: [ 141991  235298  863492  680391  746816  597521 1102539  479648  613059
  799762], Execution time: 4.3418 seconds
Query 6: Nearest neighbors distances: [0.32491714 0.33902144 0.34909308 0.363257   0.37112522 0.39377868
 0.39402556 0.40830952 0.41055548 0.4143436 ]

 80%|████████  | 8/10 [00:34<00:08,  4.36s/it]
Query 7: Nearest neighbors indices: [126610 692778 390210 389934 683747 343628   1231 411697 635087 352606], Execution time: 4.3497 seconds
Query 7: Nearest neighbors distances: [0.46419632 0.46616632 0.46887124 0.4706285  0.47223818 0.47258615
 0.4728161  0.47313237 0.4740094  0.48028678]

 90%|█████████ | 9/10 [00:39<00:04,  4.37s/it]
Query 8: Nearest neighbors indices: [ 282309 1143087  839330  867615  402817   10347  787117  583277  379662
  356535], Execution time: 4.3904 seconds
Query 8: Nearest neighbors distances: [0.22978008 0.32861102 0.36337483 0.4038509  0.4228289  0.4288115
 0.43194228 0.43435186 0.4433967  0.4447456 ]

100%|██████████| 10/10 [00:43<00:00,  4.37s/it]
Query 9: Nearest neighbors indices: [505602 646337 317731 797071 496427 226657 493742 575987  44935 391014], Execution time: 4.4466 seconds
Query 9: Nearest neighbors distances: [0.4157973  0.44511503 0.44626427 0.45290726 0.45617008 0.45865387
 0.46201754 0.46445727 0.46969557 0.47084534]
~~~~

Execution count recorded by the notebook: `7`.

## Cell 7 > code

### Prompt

Implement notebook code cell 7 of 7. Write the complete executable Python for this stage, using the existing assignment protocol and the state created by earlier cells. Include imports, helper calls, measurements, persistence, and validation required by this cell; do not replace implementation with pseudocode.

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
Query 0 | random vs true | mean abs distance diff: 0.1420
Query 1 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 1 | random vs true | mean abs distance diff: 0.1510
Query 2 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 2 | random vs true | mean abs distance diff: 0.1860
Query 3 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 3 | random vs true | mean abs distance diff: 0.1570
Query 4 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 4 | random vs true | mean abs distance diff: 0.1620
Query 5 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 5 | random vs true | mean abs distance diff: 0.1500
Query 6 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 6 | random vs true | mean abs distance diff: 0.2880
Query 7 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 7 | random vs true | mean abs distance diff: 0.1410
Query 8 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 8 | random vs true | mean abs distance diff: 0.2810
Query 9 | random vs true | index matches: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
Query 9 | random vs true | mean abs distance diff: 0.1450
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
0      0                      0.0                      0.142   
1      1                      0.0                      0.151   
2      2                      0.0                      0.186   
3      3                      0.0                      0.157   
4      4                      0.0                      0.162   
5      5                      0.0                      0.150   
6      6                      0.0                      0.288   
7      7                      0.0                      0.141   
8      8                      0.0                      0.281   
9      9                      0.0                      0.145   

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

Execution count recorded by the notebook: `8`.
