# nn for channel geometry

In an attempt to learn more about ML I decided to just jump in and try a project.
Predicting channel geometry with a simple neural network.

<img src="https://github.com/amoodie/channel_geom_nn/blob/master/demo/training/output.gif" alt="demo_gif">

I decided to use the data from Li et al., 2015 [paper link here](https://www.tandfonline.com/doi/abs/10.1080/00221686.2014.939113), which contains data for 231 river geometries.

The dataset has variable bankfull discharge, width, depth, channel slope and bed material D50 grain size.
```
             Qbf.m3s        Bbf.m          Hbf.m               S         D50.mm
count     231.000000   231.000000     231.000000      231.000000     231.000000
mean     5677.704870   234.365378       3.902396        0.003706      26.984729
std     22272.474031   538.586544       6.189606        0.007011      38.927618
min         0.337254     2.255520       0.219456        0.000009       0.010000
25%        19.113871    14.106600       0.944880        0.000287       0.400000
50%        66.000000    34.024824       1.630000        0.001490       7.330000
75%       849.505398   138.675000       4.382500        0.003600      43.000000
max    216340.707963  3400.000000      48.117760        0.052000     167.500000
```

We want to be able to predict the width, depth, and slope from the discharge and grain size alone.
This is typically a problem, because we are trying to map two input features into three output features.
In this case though, the model works because the output H and B are highly correlated. 

![correlation](https://github.com/amoodie/channel_geom_nn/blob/master/figures/scatter.png)

The network is a simple ANN, with one hidden layer with 3 nodes.

## Using/testing the model
* clone the repo
* you will need tensorflow installed
* run the main model script `channel_geom_nn_QDtoHBS.py`
* modify the content of the script to change the number of nodes, layers, normalization, optimizer, etc.
