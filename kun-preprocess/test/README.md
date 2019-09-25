
Preprocessing steps include
- Blurring (remove noise)
  - GaussianBlur
  - MedianBlur
- Deskewing 
  - Canny edge detection
  - Find border contour (one covering the largest area)
  - Four-pt transform to obtain top-down view
  - Adaptive Threshold (Binarisation)
- Plotting bbox on contours
  - Canny edge detection
  - Dilate the text
- 

```bash
python preprocess.py /path/to/img/dir/*.tif
```
