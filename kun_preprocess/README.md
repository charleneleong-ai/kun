
Preprocessing steps include
- Blurring (remove noise)
  - GaussianBlur
  - BilateralBlur
- Deskewing 
  - Canny edge detection - autocanny
  - Find border contour (one covering the largest area)
  - Four-pt transform to obtain top-down view
  - Adaptive Threshold (Binarisation) - not effective!
- Splitting into sections - very sensitive to dilation!! ><
  - Canny edge detection
  - Dilate the text
  - Draw around contour
  - Merge bboxes for n times to get sections

```bash
$ python preprocess.py /path/to/img/dir/*.tif
```
```bash
$ export GOOGLE_APPLICATION_CREDENTIALS=test/gcloud_vision/credentials.json
$ python detect_chars.py
```