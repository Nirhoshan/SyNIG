# SyNIG
This is the repository of the accepted paper [SyNIG: Synthetic Network Traffic Generation through Time Series Imaging](https://www.computer.org/csdl/proceedings-article/lcn/2023/10223392/1QdFP7dtBBK) in IEEE Local Computer Networks (IEEE LCN) 2023.

## Requirements
Following packages are required.

* Numpy				
*	Pandas			
*	matplotlib
* scipy           1.4.1
* pyts            0.12.0

## Breif overview of SyNIG

Immense growth of network usage and the associated
proliferation of network, traffic, traffic classes, and diverse
QoS requirements pose numerous challenges for network operators.
Though data-driven approaches can provide better solutions
for these challenges, limited data has been a barrier to developing
those methods with high resiliency. In this work, we propose
SyNIG (Synthetic Network Traffic Generation through Time
Series Imaging) , which utilizes Generative Adversarial Networks
(GANs) for network traffic synthesis by converting time series
data to a specific image format called GASF (Gramian Angular
Summation Field). With GASF images we encode correlation
between samples in 1D signals on a single 2D pixel map. Taking
three types of network traffic; video streaming, accessing websites
and IoT, we synthesize over 200,000 traces using over 40,000
original traces generalizing our method for different network
traffic. We validate our method by demonstrating the fidelity of
the synthetic data and applying them to several network related
use cases showing improved performance.

Overall process of SyNIG

<img src="overall_process.jpg" width="700">

## Data
We have released the data for the purpose of re-implementing and testing the algoirhtm [here](https://drive.google.com/drive/folders/1qoNrghez1vffgApGe9SnUXSzV9fx6unz?usp=sharing). This dataset is not the complete one. Complete dataset will be available upon the request.
Folder hierarchy of the current dataset,

--Platform used (e.g., YouTube, Netflix, DF)

---- Number of original traces used in post-processing steps(e.g. Traces_80)

------- Actual/GAN output/Post-processed data

------- Actual: Actual data used for the GAN model training and post-processing steps. All `.csv` files are prefrixed with the `pltform_class-ID_trace-ID`

------- GAN: generated intermediate data from the GAN model

-----------vid-x: represent one class. We have given only 2 classes (`vid` has been used as the naming convention for different classes).

---------------`platform-i.csv`: In each class folder, there are _n_ of synthetic traces from the GAN model.

------- Post: Post processed data. Each synthetic image from the GAN folder is post-processed and stored in this folder.

-----------vid-x: represent one class. We have given only 2 classes (`vid` has been used as the naming convention for different classes).

---------------`platform-i.csv`: In each class folder, there are _n_ of post-processed synthetic traces.



## Run scripts

To run the script clone the repository to your local repository and install the required packages above. 

### Run GAN model

### Run post-processing steps
For YouTube and Netflix run Run `post-processing-CSIRO.py` giving the following arguments appropriately
* `--platform`                 either YouTube or Netflix
* `--video`                    class ID. According to the dataset this should be 0 or 1
* `--no_of_ori_traces`         num of original traces used for the post-processing steps
* `--data_path`                path to the data
  
sample implementation

` python3 post-processing-CSIRO.py --platform 'YouTube'  --video 0 --no_of_ori_traces 80 --data_path <folder path> `

You can use the similar command line for `post-processing-DF.py`

## Please cite our work if you intend to use our dataset or refer to the paper.

*Sivaroopan, N., Madarasingha, C., Muramudalige, S., Jourjon, G., Jayasumana, A. and Thilakarathna, K., 2023, October. SyNIG: Synthetic Network Traffic Generation through Time Series Imaging. In 2023 IEEE 48th Conference on Local Computer Networks (LCN) (pp. 1-9). IEEE.*

`@inproceedings{sivaroopan2023synig,`\
  `title={SyNIG: Synthetic Network Traffic Generation through Time Series Imaging},`\
  `author={Sivaroopan, Nirhoshan and Madarasingha, Chamara and Muramudalige, Shashika and Jourjon, Guillaume and Jayasumana, Anura and Thilakarathna, Kanchana},`\
  `booktitle={2023 IEEE 48th Conference on Local Computer Networks (LCN)},`\
  `pages={1--9},`\
  `year={2023},`\
  `organization={IEEE}`\
`}`

### email:



