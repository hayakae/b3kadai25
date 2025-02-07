# 提案手法

## 環境
(in laplace)
- **Anaconda:**  
```
conda activate tf-gpu-env
```
- **Docker**
  
1. Build & run 
```
sudo docker build -t sakurai/comp_rec -f /data/sakurai/nitori/Compatibility_Rec/Dockerfile /data/sakurai/nitori/Compatibility_Rec
```
```
HostD=/data/sakurai/ && \
ContainerD=/data/sakurai/ && \
sudo docker run --gpus all -it \
-v "${HostD}":"${ContainerD}" \
-w "${ContainerD}" \
--shm-size=16g \
--name sakurai_comp_rec \
sakurai/comp_rec bash
```

The original code is available in the `laplace:/data/sakurai/nitori/Compatibility_Rec/PM_nitori_publish` directory.

## Run the Codes

To run the model, use the following command:

```bash
python3 model_nitori.py --train_mode [train_mode] --lambda_cl [lambda]
```

After running the scripts, the results will be saved in the `./result` directory.

### Notes:
- 