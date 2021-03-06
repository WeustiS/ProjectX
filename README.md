# ProjectX-UIUC
This repository contains the work done by students from the University of Illinois at Urbana-Champaign for Project X.
## What Did We Do?
Renewable energy generation changes are unpredictable, and are dangerous to the energy grid. In practice, we accomodate for these rapid changes in renewable energy through dispatachable energy generation, the vast majority of which is hydroelectric and natural gas. A major way to combat the supply/demand mismatch at the day is to adjust the price of energy in real-time energy markets. Two major focuses of increasing battery utilization in grids are first decreasing the price (or increasing the effectiveness) of the physical battery and second to more intelligently integrate these batteries into the grid. Our solution explores the second possibility, through reinforcement learning we should be able to redirect energy back to the grid irrespective of the size of the batteries.

## Data Acquisition
We explore California's energy grid, as it is leading the shift to renewable energy generation. The ISO wholesale power market prices electricity based on the cost of generating and delivering it from particular grid locations called nodes. We collect the price of energy for 15 minute intervals for over 30 nodes in California. We also collected the wind and solar energy forecast, as well as the demand forecast. The data collected was for the years 2018, 2019, 2020. We use the California ISO Open Access Same-time Information System (OASIS) for the acquisition. We collect the data in the format of XMLs, from which we collect the relevant information to make dataframes. We queried data a day at a time, and ran loops over months and nodes/markets. You can click https://drive.google.com/drive/folders/1zJ1xFC4Szyo1l7OT112TMWbjWyfXlMlQ?usp=sharing to view all the collected data.


## Folder Descriptions

### DataAcquisition

Scripts for pulling data from CAISO / OASIS. 


### Images

Images used in the document, as well as other assorted images for visualizations 


### Models

Saved models. Most up-to date model is TD3_Large_3


### PreprocessedData

Processed datasets in numpy format


### Training

RL training scripts


### gym_custom

Custom gym environment, utilize with pip install -e gym_custom
