# Compressive Sensing

```
March 23, 2021
```
```
Kiegan Lenihan
```


## Problem Definition

```
How do we recover an images subtleties and character when only given access to a percentage of the original pixels? If we want to preserve an images quality but reduce it’s data footprint, what is the best method?
```

## Introduction

The world’s image creation is exploding. Billions of images are created daily, and their quality is vital to our society. This presents a great opportunity to create technology that can preserve image quality with minimal space expense.

Compressive Sensing is a technique which reconstructs a sampled or corrupted signal via mathematical transformation. This technique is widely used in image compression, and reconstruction. As we will see in the results section, while parameter tuning is important for getting good results without massive overhead, compressive sensing is an effective way to reconstruct an image after sampling.

## Why Compressive Sensing?

There are many signal recovery techniques that are available today. Many involve training huge amounts of data for a model to learn how to reconstruct a signal. There are also many transformation based signal reconstruction/compression techniques; this implementation of compressive sensing will utilize the discrete cosine transform. The main motivation behind compressive sensing is that through optimization, a sparse signal can be recovered from far fewer samples required by the usual Nyquist frequency of ½ the original signal’s frequency. But images are rarely sparse, so how do we apply compressive sensing to image reconstruction?

## Discrete Cosine Transform

We can transform a non-sparse image into a sparse vector via the Discrete Cosine Transform (DCT). This transforms the image _g(x, y)_  into a sparse DCT Coefficient vector _G(u, v)_  via the DCT operator. This leads to a system of the form C = Tα. Now we have the same image in a new domain such that the signal is sparse!

## What can we do with the DCT?

We showed how easily we can go from an image pixel to a DCT coefficient, but what about the other way around? The DCT operator is invertible, so we can similarly convert a vector of DCT coefficients into image pixels with the inverse DCT transformation. This allows recovery of a full image from a sparse vector of DCT coefficients! But what happens if we don’t have every pixel? Can we still
convert the image vector into DCT coefficients?


## Sampling

To simulate an image being corrupted or compressed, a random selection of pixels is removed from the original image, and the corresponding rows are removed from the DCT operator. Note that finding the DCT coefficients is significantly more difficult, because now, we have an indeterminate system of the form B = Aα. Whereas before we could solve for α with Numerical Linear Algebra, now we must estimate the DCT coefficients with an optimization method. Enter Lasso Minimization. For the indeterminate system of the form B = Aα, we solve the regression problem via Lasso with L 1 minimization:

## Optimization

We choose the L 1 norm regularization because L 1 regularized solutions are typically sparse. This is due to the nature of the L 1 norm of a vector, which will grow greatly as the vector gets more non-zero entries. This means that we can force our solution α to be sparse, which is an appropriate assumption for the form of the DCT coefficients.

## Calculating λ
Lasso requires a regularization parameter λ, but how is it picked? When performing Lasso minimization on a set of _S-m_ training pixels, we get _S_ DCT coefficients because we only remove the rows of the DCT operator _A_ , therefore, the dimensions of α remain the same no matter how many pixels we remove. Then, we compare a selection of _m_ testing DCT coefficients calculated via Lasso with their respective DCT coefficients calculated via DCT of the original _S_ pixels. This comparison is in the form of Mean Squared Error (MSE). Now, we have a function of MSE in terms of λ. Thus, by passing _n_ values of λ, we can find _n_ MSE’s, and choose the λ which results in the lowest MSE. To ensure λ is chosen correctly, we validate the function MSE(λ) via Cross Validation. The two cross validation methods used in this project are K-folds and Random Subset.

## K-Folds Cross Validation
In K-folds Cross Validation, each set of _S_ pixels is split into 25% test pixels, and 75% training pixels. We calculate the DCT coefficients from the training pixels, and evaluate their accuracy with the test pixels, as described on slide 9. We do this four times to ensure every pixel in _S_ is a testing pixel at least once. This method gives us the best fold to approximate the DCT coefficients of a sample without checking every single pixel.

## Random Subset Cross Validation
In Random Subset Cross Validation, a random set of _m =_ floor( _S_ /6) test pixels is chosen, with the remaining _S-m_ pixels representing the training set. We repeat this process twenty times per sample and validate λ by calculating the MSE of each sample as described in slide 9. This should work better for lower values of _S_ , but may be flawed because a given pixel may never be chosen as a test pixel or a training pixel, while every pixel is guaranteed to be both at least once for K-folds CV.

## Median Filtering

Median filtering is the process of replacing a pixel with the median of its neighborhood. This process is particularly good at removing noise from an image, much better than mean or high/low/bandpass filtering because the middle value is taken to replace each pixel. This is appropriate for compressive sensing via Lasso minimization because outliers are common after the L 1 norm regularization and inverse DCT. Median filtering is used after the image is reconstructed to further improve quality.

## Compressive Sensing in Practice
In practice, we split up the image into blocks, to make it easier to calculate the DCT coefficient matrix, and perform Lasso Minimization. For each block, we perform an image reconstruction simulation, whose parameters are: filtering method (median filtering or not), cross validation method (K-folds or random), block size and sampling pixels per block.

## Results
For each block, the MSE was calculated and the log of MSE was plotted. Interestingly enough, the MSE is greatest at locations of large contrast. The MSE image files themselves give a good comprehendible interpretation of the image because they are higher (red) around the areas of high contrast and lower (blue) around the area of low contrast.

MSE is a good metric for determining image quality. The simulations with lower MSE look more detailed than those with higher MSE, when comparing the same picture. However, when comparing two different pictures, sometimes MSE is not the best metric for determining image quality. For example, the two images can have similar MSEs while having different image quality. This may be due to the different block sizes, but all of the other simulation parameters are the same. The key difference in these 
 two images is that lena.bmp is an RGB image, so the DCT coefficients are calculated for each channel and then stitched back together for the image reconstruction. This process may be contributing some high-level form of regularization that is beyond the scope of this report.

## Discussion (MSE error by Simulation Parameter)
The application of the median filter is the most significant parameter for lowering MSE. Also, smaller block sizes lead to worse image reconstructions. And although there is no discernible difference between K-folds and Random CV for simulations with median filtering, K-folds is clearly best when the reconstructed image is not median filtered.

## Discussion (Time by Simulation Parameter)
The greatest factor for determining simulation time here is more difficult to discern than in determining MSE, but in general random subset is more computational intensive than K-folds CV. Interestingly, the 32 x 32 block size simulation is quicker than all other block sizes until more than 20% of pixels are sampled, then the computational cost explodes. Conversely, all of the other block sizes actually decrease in computational time as the sample rate goes up.

The simulation time most greatly depends on how many times Lasso minimization has to happen, and how long each call to Lasso. Clearly since there are so many blocks for lena.bmp in the 4 x 4 pixel block size case (128^2 ), the amount of Lasso minimizations is very high. However, when the number of sampled pixels is sufficiently large, as is the case for a block size of 32 x 32, the time to complete each Lasso minimization is most significant. This is why the higher sampling rates for 32 x 32 blocks take longer: the size of the undetermined system in slide 7 grows exponentially. Other than the 32 x 32 case, random CV increases computation time because the number of Lasso simulations is five times greater than K-folds CV.

## Final Thoughts
In general, the higher sample rates resulted in better image approximations, which is intuitive. The median filtering also greatly improved image reconstructions, because outliers with unusually high or low values from the Lasso minimization or inverse DCT transformation are normalized.

Greater block size improved the image approximations, and is less computationally complex until the block size becomes too great, as was the case with the 32 x 32 block size for lena.bmp. It should be noted however that parallelization was not included in these simulations, and the language used was interpretive (python) so perhaps a more appropriate linear algebra solver or compiled language would buck the inconsistency in simulation time for the 32 x 32 block.

Lastly, K-Folds outperformed Random Cross Validation in all cases, including the low sample size examples, which was unexpected.

## References
● SkLearn for Lasso Minimization: API design for machine learning software: experiences from the
scikit-learn project, Buitinck et al., 2013
● NumPy: Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau,
D., ... Oliphant, T. E. (2020). Array programming with NumPy. Nature, 585, 357–362.
https://doi.org/10.1038/s41586-020-2649-2
● SciPy: Virtanen, P., Gommers, R., Oliphant, T. E., Haberland, M., Reddy, T., Cournapeau, D., ...
SciPy 1.0 Contributors. (2020). SciPy 1.0: Fundamental Algorithms for Scientific Computing in
Python. Nature Methods, 17, 261–272. https://doi.org/10.1038/s41592-019-0686-2
● MatPlotLib: Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science
&amp; Engineering, 9(3), 90–95.
● OpenCV: Bradski, G. (2000). The OpenCV Library. Dr. Dobb&#x27;s Journal of Software Tools.
● Pillow Image Library: Clark, A. (2015). Pillow (PIL Fork) Documentation. readthedocs. Retrieved
from https://buildmedia.readthedocs.org/media/pdf/pillow/latest/pillow.pdf
● Liu, H. H. (2019). Machine learning a quantitative approach. California: PerfMath.

## Collaborations
● Hadiya Harrigan, Max Nedzydur and Thought Pod 1 on Sakai
