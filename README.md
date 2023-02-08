# sf_segmenter
---
Music Segmentation/Labeling Algorithm. Based on SF (Structural feature) method.   
Modified from [MSAF](https://github.com/urinieto/msaf). I simplified the IO for arbitrary input features and quick experiments. 

## Installation 
version: 0.0.1
```
pip install sf-segmenter
```

To use `sf_segmenter.vis`, matplotlib needs to be installed; you can install it with the `vis` extra:

```
pip install sf-segmenter[vis]
```


## Reference
* Serrà, J., Müller, M., Grosche, P., & Arcos, J. L. (2012). Unsupervised Detection of Music Boundaries by Time Series Structure Features. In Proc. of the 26th AAAI Conference on Artificial Intelligence (pp. 1613–1619).Toronto, Canada.

* J. Serrà, M. Müller, P. Grosche and J. L. Arcos, "Unsupervised Music Structure Annotation by Time Series Structure Features and Segment Similarity," in IEEE Transactions on Multimedia, vol. 16, no. 5, pp. 1229-1240, Aug. 2014, doi: 10.1109/TMM.2014.2310701.

## Resources

* audiolabs/FMP course/Chapter 4: Music Structure Analysis: https://www.audiolabs-erlangen.de/resources/MIR/FMP/C4/C4.html

