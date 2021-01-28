package tabnet

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

// Embedding implements a embedding layer
func Embedding(m *Model, embeddingSize int, embeddingDim int, opts EmbeddingOpts) Layer {
	lt := incLayer("Embedding")

	if opts.WeightsInit == nil {
		opts.WeightsInit = gorgonia.Gaussian(0.0, 1.0)
	}

	w := m.addWeights(lt, tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, *gorgonia.Node, error) {
		err := m.checkArity(lt, inputs, 1)
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
