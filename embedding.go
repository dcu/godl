package tabnet

import (
	"log"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type EmbeddingOpts struct {
	WeightsInit gorgonia.InitWFn
}

// Embedding implements a embedding layer
func (m *Model) Embedding(embeddingSize int, embeddingDim int, opts EmbeddingOpts) Layer {
	w := m.addWeights(tensor.Shape{embeddingSize, embeddingDim}, opts.WeightsInit)

	return func(inputs ...*gorgonia.Node) (*gorgonia.Node, error) {
		err := m.checkArity("Embedding", inputs, 1)
		if err != nil {
			return nil, err
		}

		indices := inputs[0]
		indicesShape := indices.Shape().Clone()
		indices = gorgonia.Must(gorgonia.Reshape(indices, tensor.Shape{indicesShape.TotalSize()}))

		log.Printf("ByIndices(%v, %v)", w.Shape(), indices.Shape())

		embedding, err := gorgonia.ByIndices(w, indices, 0)
		if err != nil {
			return nil, err
		}

		embedding = gorgonia.Must(gorgonia.Reshape(embedding, append(indicesShape, embeddingDim)))

		return embedding, nil
	}
}
