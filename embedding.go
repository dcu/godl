package godl

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

type EmbeddingModule struct {
	model                       *Model
	layer                       LayerType
	opts                        EmbeddingOpts
	embeddingSize, embeddingDim int

	weight *Node
}

func (m *EmbeddingModule) Forward(inputs ...*Node) Nodes {
	err := m.model.CheckArity(m.layer, inputs, 1)
	if err != nil {
		panic(err)
	}

	indices := inputs[0]
	indicesShape := indices.Shape().Clone()
	indices = gorgonia.Must(gorgonia.Reshape(indices, tensor.Shape{indicesShape.TotalSize()}))

	embedding, err := gorgonia.ByIndices(m.weight, indices, 0)
	if err != nil {
		panic(err)
	}

	embedding = gorgonia.Must(gorgonia.Reshape(embedding, append(indicesShape, m.embeddingDim)))

	return Nodes{embedding}
}

// Embedding implements a embedding layer
func Embedding(m *Model, embeddingSize int, embeddingDim int, opts EmbeddingOpts) *EmbeddingModule {
	lt := AddLayer("Embedding")

	if opts.WeightsInit == nil {
		opts.WeightsInit = gorgonia.Gaussian(0.0, 1.0)
	}

	w := m.AddWeights(lt, tensor.Shape{embeddingSize, embeddingDim}, NewWeightsOpts{
		InitFN: opts.WeightsInit,
	})

	return &EmbeddingModule{
		model:         m,
		layer:         lt,
		opts:          opts,
		embeddingSize: embeddingSize,
		embeddingDim:  embeddingDim,
		weight:        w,
	}
}
