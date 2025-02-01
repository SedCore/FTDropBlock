# FT-DropBlock
Implementation of Features-Time DropBlock (FT-DropBlock) regularization strategy for Electroencephalography(EEG)-based Convolutional Neural Networks (CNNs).

## Abstract
The use of Deep Learning (DL) and digital signal processing in Brain-Computer Interfaces (BCIs) to improve the analysis of Electroencephalogram (EEG) data has shown promise, but overfitting of DL models remains a challenge, particularly in Convolutional Neural Networks (CNNs) designed for spatiotemporal data. This paper introduces Features-Time DropBlock (FT-DropBlock), a novel adaptation of DropBlock regularization tailored for EEG processing, targeting the spatiotemporal dimensions represented by features and time points to achieve structured regularization. Contiguous blocks of features and their associated time points are strategically dropped, promoting robust learning by encouraging spatiotemporal coherence and reducing overfitting. To our knowledge, at the time of writing, no previous use of DropBlock in EEG-based CNNs has been reported in the literature. Our approach is the first to use DropBlock to enforce structured regularization across features and time dimensions to preserve spatial consistency within feature maps and ensure temporal regularization. We tested our approach by replacing conventional Dropout with FT-DropBlock at selected regularization stages in three EEG-based CNNs. Experimental results conducted on the publicly available BCI Competition IV 2a Dataset show that our approach demonstrates significant improvements over traditional Dropout regularization in classification accuracy and robustness against overfitting, highlighting the effectiveness of this targeted FT-DropBlock strategy for EEG-based CNNs.

## Prerequisites
We ran our experiment in the following environment:
* ● &ensp; NVIDIA Tesla T4 GPU with 16GB RAM
* ● &ensp; Linux Ubuntu 24.04 LTS operating system
* ● &ensp; CUDA 12.6 library
* ● &ensp; Python 3.12.3 x64
* ● &ensp; Tensorflow 2.18.0 with XLA compiler

## Usage
### Models tested
* ● &ensp; EEGNet v4 [(GitHub repo)](https://github.com/vlawhern/arl-eegmodels). Original paper [here](https://doi.org/10.1088/1741-2552/aace8c).
* ● &ensp; EEG-TCNet [(GitHub repo)](https://github.com/iis-eth-zurich/eeg-tcnet). Original paper [here](https://doi.org/10.1109/SMC42975.2020.9283028).
* ● &ensp; EEG-ITNet [(GitHub repo)](https://github.com/AbbasSalami/EEG-ITNet). Original paper [here](https://doi.org/10.1088/1741-2552/aace8c)

### Dataset
We used the [BCI Competition IV 2a](https://www.bbci.de/competition/iv) dataset, imported in our code using the [MOABB](https://doi.org/10.5281/zenodo.10034223) library.

### Running the models
Run the main.py file. The use of parameters is optional since there are default values, unless you need to choose other values.
```
python3 main.py --model=EEGNetv4
```
Parameters:
* --model: CNN model. Choices: EEGNetv4 (default), EEGTCNet, EEGITNet.
* --reg: Regularization method. Choices: 0 for Dropout or 1 for FT-DropBlock (default).
* --prob: Overall drop probability: 0.2, 0.3, 0.4, 0.5 (default), 0.6, 0.7, 0.8, 0.9
* --block: Block size value for FT-DropBlock: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 (default), etc.

## Paper citation
If you use FT-DropBlock regularization in your research and found it helpful, please cite our paper:
```
@inproceedings{Sedi2025ftdropblock,
  author={Sedi Nzakuna, Pierre and Paciello, Vincenzo and Gallo, Vincenzo and Lay-Ekuakille, Aimé and Kuti Lusala, Angelo},
  title={FT-DropBlock: A Novel Approach for SPatiotemporal Regularization in EEG-based Convolutional Neural Networks},
  booktitle={2025 IEEE International Instrumentation and Measurement Technology Conference (I2MTC)},
  year={2025},
  doi={}
}
```
## License
Please refer to the LICENSE file.
