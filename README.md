
### Docker Commands

To build the image:

```bash
docker build . -t cpp_nn
```

To start the image:

```bash
docker run -itd --name cpp_nn -v $(pwd):/app cpp_nn
```