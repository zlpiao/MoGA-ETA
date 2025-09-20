<p align="center">

  <h2 align="center"><strong>MoGA-ETA: Generalized Face Anti-Spoofing with
Enhanced Text Guidance and Alignment</strong></h2>

</p>


<p align="center">
</p>



##  Updates :
- 09-09-2025: Inference code has been released.
- 20-09-2025: On Baidu Netdisk, share the optimal models based on Benchmark-1 and Benchmark-2.
- The training code will be made publicly available on GitHub.




## Instruction for code usage :

### Setup
- Get Code
```shell
 git clone https://github.com/zlpiao/MoGA-ETA.git
```
## Inference
```shell
cd MoGA-ETA
python infer.py \
        --report_logger_path path/to/save/performance.csv \
        --ckpt best_model/checkpoint.pth \
        --config O \
        --method moga_eta
```

## Training moga_eta
```shell
cd MoGA-ETA
python train_moga_eta.py  \
       --op_dir output/MoE/   \
       --report_logger_path path/to/save/performance.csv    \
       --config O  \
       --method moga_eta 
```
TODO: Integrate all the training scripts into one file and add configurable arguments to choose the method and benchmark more easily.

## Citation
If you're using this work in your research or applications, please cite using this BibTeX:
To Be Completed
```bibtex
  @InProceedings{-----,
    author    = {Liepiao Zhang, Kun Liu, Junduan Huang, Zitong Yu, Wenxiong Kang},
    title     = {MoGA-ETA: Generalized Face Anti-Spoofing with Enhanced Text Guidance and Alignment},
    booktitle = {-------},
    month     = {October},
    year      = {2025},
    pages     = {------}
}
```
## File Shared via Cloud Storage:    Benchmark-1-bestmodel and   Benchmark-2-bestmodel
Benchmark-1-bestmodel.zip
Link: https://pan.baidu.com/s/1uDlUR_5-SjC9eLYnS9KamQ?pwd=42um extraction code: 42um 

Benchmark-2-bestmodel.zip
Link: https://pan.baidu.com/s/1mtCfPB4NjgTmIftEtjOaUA?pwd=7wy5 extraction code: 7wy5 


## Acknowledgement :pray:
Our code is built on top of the [FLIP](https://github.com/koushiksrivats/FLIP) repository. We thank the authors for releasing their code.
