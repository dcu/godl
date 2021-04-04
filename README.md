# GoDL

godl is **Go** **D**eep **L**earning framework written on top of Gorgonia.  
godl is to Gorgonia what Keras is to TensorFlow.

## API Stability
The API is not stable and can change at any moment.
I'm writing this framework mostly to learn and so I don't provide any guarantees
that it'll work for you. Use it at your own risk.

## Roadmap

The following items are in the current roadmap, some of them need to be implemented in Gorgonia first.

- [x] Data loader  
- [x] Base storage (save/load)  
- [ ] CLI to scaffold a project  
- [x] Embeddings  
- [ ] Publish pre-trained weights

### Losses
- [x] Cross Entropy
- [x] MSE
- [ ] CTC Losses 

### Pooling
- [x] MaxPool
- [x] AvgPool
- [ ] GlobalMaxPool
- [x] GlobalAvgPool

### Normalization
- [x] Batch Norm
- [x] Ghost Batch Norm
- [ ] GroupNorm  
- [ ] LayerNorm  

### Recurrent Layers
- [x] LSTM  
- [ ] Bidirectional  
- [ ] GRU
- [ ] ConvLSTM2D

### Reshaping
- [ ] ZeroPadding
- [ ] UpSampling

### Convolutional
- [x] Conv2D
- [ ] DepthWiseConv2D

### Applications
- [x] TabNet  
- [x] VGG16  
- [ ] VGG19  
- [ ] ResNet50  
- [ ] ResNet101 
- [ ] VGGFace2
- [ ] YOLO  
- [ ] BERT

### Future
- [ ] Support ONNX  
- [ ] Support hdf5 files  
