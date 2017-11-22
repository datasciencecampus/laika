# Skynet notes

hold back 100 images for testing

```
# randomise samples
cat sample-filtered.txt |gsort -R > samples.random

# test samples
cat samples.random |head -100 > test-sample-filtered.txt

# training samples
cat samples.random |tail -n +101 > sample-filtered.txt
```
