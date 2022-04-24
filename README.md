# LawyerPAN: A Proficiency Assessment Network for Trial Lawyers

The code is the implementation of LawyerPAN, and the dataset is collected from the 51djl.com.



## Usage

Run divide_data.py to divide the original data set data/log_data.json into train set, validation set and test set:

`python divide_data.py`

Train the model:

`sh start.sh`



## Dataset

The LCHR-small dataset has been uploaded to Baidu Cloud 

`https://pan.baidu.com/s/1rvzhPgwQHBVC6UVeba8mwQ TOKEN：g2o2`

- log_data.json: The log data is organized in the structure:

`{"user_id": user_id, "log_num": log_num, "logs": [log1, log2, ...]]}`

`log = {"item_id": item_id, "score": score, "field_code": [field_code1, field_code2, ...], "tag": tag, "member": [member1, member2, ...]}`

- facv2vec.json: The case descriptions are encoded by BERT, organized as following structure：

`{"item_id": case_descriptions}`

To run the code, please place the downloaded dataset under the `data` folder.


## Citation
If this code helps with your studies, please kindly cite the following publication:

```
@inproceedings{an2021lawyerpan,
  title={Lawyerpan: A proficiency assessment network for trial lawyers},
  author={An, Yanqing and Liu, Qi and Wu, Han and Zhang, Kai and Yue, Linan and Cheng, Mingyue and Zhao, Hongke and Chen, Enhong},
  booktitle={Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery \& Data Mining},
  pages={5--13},
  year={2021}
}
```
