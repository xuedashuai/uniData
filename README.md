## Unidata 
### Introduction
  Unidata is a newly-developed data interface proposed by Xue et.al. for vehicle trajectory prediction.
  Compared with the traditional data interface, the new interface standardizes the data from different data sets, uniformly converts them into unidata format, and further adjusts the data in unidata format to the network to be tested.
  Assuming that there are X methods that need to be tested on Y data sets, the traditional method needs to design X times Y data interfaces, and unidata only needs to design X plus Y data interfaces.
  Unidata currently supports five data sets commonly used in trajectory prediction, and has completed tests on cslstm, mfp, pishgu, and hivt. Support for forecast-mae, and qcnet will be released as they are organized.
  Unidata uses the uni-batch strategy to alleviate memory consumption by reducing the data capacity of each batch. Each uni-batch in Unidata contains only one target vehicle and the historical and future trajectories of its surrounding vehicles.
### Files
  Folder d2u contains data interfaces for processing raw datasets. Folder u2m contains data interfaces for sending unidata to neural networks.
### Unidata
  Unidata is stored in npy format. Shape: [N,T,F]. N:Number of vehicles in an unidata; T:Total time length; F:Numbers of features. F:[veh_id,frame_id,x,y,type,v,a,lane,flag].

