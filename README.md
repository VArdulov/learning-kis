# LKIS

This is a reimplementation of the following paper using PyTorch 1.7 

Naoya Takeishi, Yoshinobu Kawahara, and Takehisa Yairi, "Learning Koopman Invariant Subspaces for Dynamic Mode Decomposition," in *Advances in Neural Information Processing Systems (Proc. of NIPS)*, vol. 30, pp. 1130-1140, 2017.

arXiv preprint: <https://arxiv.org/abs/1710.04340>

## Requirements
This code base was built with the following libraries and environments:
1. `python==3.8.5`
2. `numpy==1.20.0`
3. `scipy==1.6.0`
4. `torch==1.7.1`
5. `pandas==1.2.1`

For visualizations we'll use :
1. `matplotlib==3.3.4`
2. `seaborn==0.11.1`

This environment can be installed easily with the following command:
```bash
pip install -r requirements.txt
```
## Files

* `lkis.py`
	- Core implementation of LKIS network.
* `train.py`
	- Script for training network.
* `predict.py`
	- Script for test by prediction based on a trained model.
* `exp_lorenz`
	- Root directory for experiment using Lorenz series. Dataset is included here.

### Example

```
python train.py train.py --l 7 --i 5
```

## Important options

### train.py

## Author

*  **Naoya Takeishi** - [http://www.naoyatakeishi.com/](http://www.naoyatakeishi.com/)
* **Victor Ardulov** - [https://Vardulov.github.io](https://Vardulov.github.io)

## License

This project is licensed under the MIT License - see the [LICENSE.txt](LICENSE.txt) file for details