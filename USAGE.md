
### Preparation

1. Install TensorFlow >= 1.7

2. Follow steps 1-4 in [facenet wiki](https://github.com/davidsandberg/facenet/wiki/Validate-on-LFW)

3. Clone this repo and uncompress the pre-trained models inside:
```bash
git clone https://github.com/ppwwyyxx/Adversarial-Face-Attack
cd Adversarial-Face-Attack
wget
tar xjvf model-20180402-114759.tar.bz2
```
You can also [download the model from facenet](https://github.com/davidsandberg/facenet#pre-trained-models).

4. Validate models and the dataset:
```
./face_attack.py --data /path/to/lfw_mtcnnpy_160 --validate-lfw
# /path/to/lfw_mtcnnpy_160 is obtained above in step #4 in facenet wiki.
```

### Run attack

```bash
./face_attack.py --data /path/to/lfw_mtcnnpy_160 \
	--attack images/clean-JCJ.png --target Arnold_Schwarzenegger --output JCJ-to-Schwarzenegger.png
```
