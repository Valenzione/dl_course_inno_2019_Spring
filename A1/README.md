Docker image is available as [`valenzione/y-gavrilin-a1:latest`](https://cloud.docker.com/u/valenzione/repository/docker/valenzione/y-gavrilin-a1) (click to visit DockerHub).

To build locally:
 ```
docker build . -t y-gavrilin-a1   
```


To train:
 ```
docker run -it --mount type=bind,source="$(pwd)"/data,target=/app/data  y-gavrilin-a1 --mode train
```
Trained model will be saved as `data/output/model` binary.

To predict:
 ```
docker run -it --mount type=bind,source="$(pwd)"/data,target=/app/data  y-gavrilin-a1 --mode predict
```
Images should be placed in `data/input`. Prdockedicted classes will be printed to console and saved in `data/out/results.txt`