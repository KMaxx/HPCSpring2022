**Build:**
```
mkdir build;
cd build;
cmake ..;
ln -s ../images/* .
make
```
or
```
mkdir build;
cd build;
ln -s ../images/* .
make -f ../Makefile;
```

**Run:**
```
./rt [-s num_of_samples_per_pixel (default: 100)] [-d max_depth (default: 50)]
```

**Example:**
```
./rt -s 100 -d 5 > output.ppm
convert output.ppm output.png
```

