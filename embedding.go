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
	lt := AddLayer("Embedding")

	if opts.WeightsInit == nil {
		opts.WeightsInit = gorgonia.Gaussian(0.0, 1.0)
	}

	w := m.AddWeights(lt, tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)

	return func(inputs ...*gorgonia.Node) (Result, error) {
		err := m.CheckArity(lt, inputs, 1)
		if err != nil {
			return Result{}, err
		}

		indices := inputs[0]
		indicesShape := indices.Shape().Clone()
		indices = gorgonia.Must(gorgonia.Reshape(indices, tensor.Shape{indicesShape.TotalSize()}))

		embedding, err := gorgonia.ByIndices(w, indices, 0)
		if err != nil {
			return Result{}, err
		}

		embedding = gorgonia.Must(gorgonia.Reshape(embedding, append(indicesShape, embeddingDim)))

		return Result{Output: embedding}, nil
	}
}
