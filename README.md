## Prepare dataset


### 1. Convert DICOM to jpg images

```
$ python batch.py cache --src data/DICOM --dest data/images
```

### 2. Extract mask from GIMP .xcf files

```
$ python batch.py extract-mask --src data/draw --dest data/mask
```

### 3. Crop plain/enhance regieon from US images.

```
$ python batch.py crop --src data/images --dest data/crop
```
