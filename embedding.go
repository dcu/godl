package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

// Embedding implements a embedding layer
func (m *Model) Embedding(embeddingSize int, embeddingDim int, opts EmbeddingOpts) Layer {
	layerType := "Embedding"

	w := m.addWeights(layerType, tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)
	// w := gorgonia.NewTensor(m.g, tensor.Float64, 2, gorgonia.WithShape(embeddingSize, embeddingDim), gorgonia.WithInit(gorgonia.GlorotN(1.0)))

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		err := m.checkArity(layerType, inputs, 1)
		if err != nil {
			return nil, nil, err
		}

		indices := inputs[0]
		indicesShape := indices.Shape().Clone()
		indices = gorgonia.Must(gorgonia.Reshape(indices, tensor.Shape{indicesShape.TotalSize()}))

		embedding, err := gorgonia.ByIndices(w, indices, 0)
		if err != nil {
			return nil, nil, err
		}

		embedding = gorgonia.Must(gorgonia.Reshape(embedding, append(indicesShape, embeddingDim)))

		return embedding, nil, nil
	}
}
