# Skynet notes

hold back 100 images for testing

```bash
cd data
mkdir test-images test-labels
```

test images
```bash
cd data/images
# randomly select 100 files, move to .../test-images.
for i in $(ls -al *.jpg |awk '{print $9}' |gsort -R |head -100) ;do mv -v $i ../test-images ;done
```

test labels
```bash
cd ../labels/color
# for each file in test-images, move it's .png equivalent label/mask to ../../test-labels.
for i in $(ls -al ../../test-images/*.jpg |awk '{print $9}' |sed 's/^.*\///; s/jpg/png/') ;do mv -v $i ../../test-labels/ ;done
```
