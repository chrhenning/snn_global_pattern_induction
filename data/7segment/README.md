# LED Display Domain dataset

A simple toy dataset to test the implementations.

The dataset has been retrieved from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/LED+Display+Domain)

> Lichman, M. (2013). UCI Machine Learning Repository [http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, School of Information and Computer Science.

## Dataset Retrieval

In Linux, the dataset can betrieved as follows:

```
wget http://archive.ics.uci.edu/ml/machine-learning-databases/led-display-creator/led-creator.names -O description.txt
wget http://archive.ics.uci.edu/ml/machine-learning-databases/led-display-creator/led-creator.c
```

This two commands give you a dataset description plus a file to generate the actual dataset.

At first, the dataset generation program has to be compiled.

```
gcc led-creator.c -o led-creator
```

Finally, the dataset can be generated as desired.

```
./led-creator numtrain seed outputfile noise
```

For instance:

```
./led-creator 1000 42 7segment.txt 1
```
